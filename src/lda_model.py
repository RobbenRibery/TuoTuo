from typing import List, Dict
import warnings 

import torch as np 
import numpy as np 

from scipy.special import psi, gammaln, logsumexp 

from src.utils import (
    get_vocab_from_docs, 
    get_np_wct, 
    expec_log_dirichlet,
    log_gamma_sum_term, 
    np_clip_for_exp,
)

from src.cutils import _dirichlet_expectation_2d

from pprint import pprint

SEED = 42 
DTYPE = float

#Dim Conventions 
#Alpha: Topic Prior, 1D NP array 
#Eta: Word Prior, 1D NP array 
def expec_real_dist_minus_suro_dist(
    expec_log_var: np.ndarray, 
    real_dist_prior:np.ndarray,
    suro_dist_prior:np.ndarray,
    num_topic: int,
) -> float: 
    
    """Pass in the num_topic here, means assuming that we let alphda being an exchangable Dirichlet as well

    Returns:
        _type_: _description_
    """
    assert expec_log_var.ndim == 2
    delta_loss = 0 

    delta_loss += np.sum((real_dist_prior-suro_dist_prior)*expec_log_var)
    delta_loss += np.sum(gammaln(suro_dist_prior)-gammaln(real_dist_prior))
    delta_loss += np.sum(gammaln(real_dist_prior*num_topic) - gammaln(np.sum(suro_dist_prior,1)))

    return delta_loss


class LDASmoothed: 

    """Class implemented the Smoothed Version of Latent Dirichlet Allocation

    - Parameters init 
    - npaining 
    - Inference 

    for smoothed version of Latent Dirichlet Allocation 
    """

    def __init__(
            self, 
            docs: List[List[str]],
            num_topics: int,
            word_ct_dict: dict,
            num_doc_population: int, 
            word_ct_array: np.ndarray = None,
            word_to_idx: dict = None, 
            idx_to_word: dict = None, 
            verbose: bool = True,
        ) -> None:

        assert get_vocab_from_docs(docs) == word_ct_dict
        
        self.D_population = num_doc_population
        self.docs = docs 
        
        # number of documents 
        self.M = len(docs)

        # number of unique words in corpus 
        if word_ct_array is not None:
            assert word_ct_array.shape[0] == len(word_ct_dict)

            val_word_ct_array, val_word2idx = get_np_wct(word_ct_dict, docs)
            assert (word_ct_array == val_word_ct_array).all()

            word_2_idx = val_word2idx 
        else:
            word_ct_array, word_2_idx = get_np_wct(word_ct_dict, docs)

        self.word_ct_array = word_ct_array

        self.word_2_idx = word_2_idx 
        self.word_2_idx:dict 

        self.idx_2_word = {v:k for k,v in self.word_2_idx.items()}
        self.idx_2_word: dict

        # number of unique words in the corpus 
        self.V = word_ct_array.shape[0]  

        # number of topics 
        self.K = num_topics

        #parameters init
        # define the DGM hyper-parameters
        # Dirichlet Prior 
        np.random.seed(SEED)
        self._alpha_ = 1#np.random.gamma(shape = 100, scale = 0.01, size =self.K)

        if verbose:
            print(f"Topic Dirichlet Prior, Alpha")
            print(self._alpha_)
            print() 
        # Dirichlet Prior - Exchangeable Dirichlet
        self._eta_ = 1

        if verbose:
            print(f"Exchangeable Word Dirichlet Prior, Eta ")
            print(self._eta_)
            print()


        # define the Convexity-based Varitional Inference hyper-parameters 
        #Dirichlet Prior, Surrogate for _eta_ 
        np.random.seed(SEED)
        self._lambda_ = np.random.gamma(shape=100, scale=0.01, size=(self.K, self.V),)
        if verbose: 
            print(f"Var Inf - Word Dirichlet prior, Lambda")
            print(self._lambda_.shape)
            print()

        #Dirichlet Prior, Surrogate for _alpha_ 
        self._gamma_ = self._alpha_ + np.ones((self.M,self.K), dtype=DTYPE) * self.V / self.K 
        if verbose: 
            print(f"Var Inf - Topic Dirichlet prior, Gamma")
            print(self._gamma_.shape)
            print()


    def approx_elbo(
            self,
            X:np.ndarray,
            sampling:bool = False, 
        ) -> float:

        """Approximate the Var inference ELBO 

        X: Input matrix: 

            - Row: document index
            - Column: vocab indxies 
            - Entries: number of time each vocab appeared in a particular document

        Returns:
            float: estimated elbo
        """

        # handlder for just one document 
        num_doc = 1 if X.ndim == 1 else X.shape[0]

        # under surogate distribution: 
        # \gamma -> \theta
        expec_log_theta = _dirichlet_expectation_2d(self._gamma_.astype(DTYPE))

        # \lambda -> \beta 
        expect_log_beta = _dirichlet_expectation_2d(self._lambda_.astype(DTYPE))

        elbo = 0

        # compute Eq[logp(z|\theta)] - Eq[logq(z|\phi)] + Eq[logp(w|beta,z)]
        for d in range(num_doc): 
            
            # indicies of vocab in the document 
            d_word_ids = np.nonzero(X[d,:])[0]
            d_word_cts = X[d,d_word_ids]

            log_phi_sum = np.zeros(len(d_word_ids), dtype = DTYPE)

            for i in range(len(d_word_ids)): 
                
                # temp in R K
                temp = expec_log_theta[d,:] + expect_log_beta[:, d_word_ids[i]]
                log_phi_sum[i] = logsumexp(temp)

            elbo += np.dot(d_word_cts, log_phi_sum)
        
        # compute Eq[log p(theta|alpha)] - Eq[log q(theta|gamma)]
        elbo += \
            expec_real_dist_minus_suro_dist(
                expec_log_var= expec_log_theta, 
                real_dist_prior= self._alpha_,
                suro_dist_prior= self._gamma_,
                num_topic= self.K 
            )
        
        # document dependent part of the loss function is completed, 
        # performe scaling 
        if sampling: 
            score = score * self.D_population/num_doc

        # document indepednet part 
        # Compute Eq[logp(\beta|\eta)] = Eq[logq(\beta|\lambda)]
        elbo += \
            expec_real_dist_minus_suro_dist(
                expec_log_var= expect_log_beta,
                real_dist_prior= self._eta_,
                suro_dist_prior= self._lambda_,
                num_topic= self.V
            )
            
        return elbo, (expec_log_theta, expect_log_beta)
    
    def approx_perplexity(self, X:np.ndarray, sampling:bool = False,) -> float: 

        """compute the perplexity (per document) based on the approximated ELBO

        """
        num_doc = 1 if X.ndim == 1 else X.shape[0]
        elbo, suff_stats = self.approx_elbo(X, sampling)
        print(elbo)

        if sampling: 
            total_word_count = np.sum(X) *  self.D_population/num_doc
        else:
            total_word_count = np.sum(X)

        temp = -elbo/total_word_count
        

        return np.exp(np_clip_for_exp(temp)), suff_stats


    def e_step_batch(
        self, 
        X:np.ndarray, 
        expec_log_theta: np.ndarray,
        expec_log_beta:np.ndarray,
        step:int = 100, 
        threshold:float = 1e-05, 
        verbose:bool = False,
    ) -> None: 
        
        """
        Update the document Var Inf parameters following the Eexpectation step of EM

        E-step: 
            - update __phi__ and __gamma__ in the direction of the partial derivative 

        """
        if verbose: 
            perplexity, suff_stats = self.approx_perplexity(X)
            print(f'Before Estep perplexity = {perplexity}')

        for d in range(self.M): 

            delta_gamma_d =  np.full(self._gamma_.shape[1], fill_value=np.inf)
            l1_delta_gamma_d = np.linalg.norm(delta_gamma_d, ord=1)

            d_word_ids = np.nonzero(X[d,:])[0]
            d_word_cts = X[d, d_word_ids, np.newaxis]

            expec_log_theta_d = expec_log_theta[d,:,np.newaxis].reshape(self.K,1) #k,1
            expec_log_beta_d = expec_log_beta[:, d_word_ids] #k, v

            it = 0 
            while l1_delta_gamma_d > threshold and it <= step:

                gamma_d = np.copy(self._gamma_[d])
                    
                ### --- Update Phi[d][v] --- ###
                phi_d = np.exp((expec_log_beta_d + expec_log_theta_d).T + 1e-10) #w,k

                ### Normalise Phi ###
                phi_d_norm = phi_d/np.sum(phi_d, axis=1)[:, np.newaxis]
                
                ### --- Update Gamma[d] --- ###
                self._gamma_[d,:] = (self._alpha_ + np.dot(phi_d_norm.T, d_word_cts)).ravel()
                expec_log_theta_d = _dirichlet_expectation_2d(self._gamma_[d,:,np.newaxis])

                ### --- update stop creteria --- ###
                delta_gamma_d = self._gamma_[d] - gamma_d 
                l1_delta_gamma_d = np.linalg.norm(delta_gamma_d, ord=1)

                it += 1 

            expec_log_theta[d,:] = expec_log_theta_d.ravel()

            if it > step:
                warnings.warn(f"Estep, Maximum iteration reached at step {it} for Document {d}")

        if verbose: 
            perplexity, suff_stats = self.approx_perplexity(X)
            print(f'After Estep perplexity = {perplexity}')

        return self._gamma_, (expec_log_theta, expec_log_beta)
            
    

    def m_step(self, step: int = 100, threshold:float = 1e-07, verbose:bool = False,) -> None: 

        self.update_alpha(step, threshold, verbose)
        self.update_eta(step, threshold, verbose)


    def update_alpha(self, step:int = 500, threshold:float = 1e-07, verbose:bool = False,) -> None: 

        """Newton-Raphson in Linear Time for the sepcial Hessian with 
        Diag(h) + 1z1T
        """

        it = 0 

        while it <= step: 

            sum_ = np.sum(self._alpha_)

            # grad in R 1*K
            g = self.M * (psi(sum_)-psi(self._alpha_)) + \
                np.sum(psi(self._gamma_), axis=0) - \
                np.sum(psi(np.sum(self._gamma_, axis=1)))
            
            # hessian diagonal vector in R 1*K 
            h = - self.M * polygamma(1, self._alpha_)

            # hessian constant part 
            z = self.M * polygamma(1, np.sum(self._alpha_))

            # offset c 
            c = np.sum(g/h) / ((1/z)+np.sum(1/h))

            # newton step s
            update = (g-c)/h 

            alpha_new = self._alpha_ - update 

            if (alpha_new < 0).any(): 
                raise ValueError(f"Negative dirichlet parameter encoutered at iteration {it}, alpda new: {alpha_new}")

            delta = np.linalg.norm(alpha_new-self._alpha_) 

            if verbose: 
                # elbo = compute_elbo(
                #     self._gamma_,
                #     self._phi_,
                #     self._lambda_,
                #     self._alpha_,
                #     self._eta_,
                #     self.word_ct_array
                # )
                print(f"M Step: Iteration {it}, Delta Alpha = {delta}")
                print(f"Alpha Old:{self._alpha_} -> Alpha New:{alpha_new}")


            self._alpha_ = alpha_new
            it += 1 

            if delta < threshold: 
                return 
            
        warnings.warn(f"Update alphda Maximum iteration reached at step {it}")

    def update_eta(self, step:int = 1000, threshold:float = 1e-09, verbose:bool = False) -> None:

        """
        Newton-Raphson in Linear Time 

        """
        
        it = 0 
        while it <= step: 
            
            # gradient 
            g = self.K * (self.V * psi(self.V * self._eta_) - self.V * psi(self._eta_)) + \
                np.sum(psi(self._lambda_))- \
                np.sum(self.V * psi(np.sum(self._lambda_, axis=1)))

            # h hessian diagonal 
            h = self.K * (
                self.V**2 * polygamma(1, self.V * self._eta_) - \
                self.V * polygamma(1,self._eta_)
            )

            # newton step 
            update = g/h

            eta_new = self._eta_ - update 
            if eta_new < 0: 
                raise ValueError(f"Dirichlet Parameter become < 0 at iteration {it}")
            print(f"Eta Old {self._eta_} -> Eta New {eta_new}")

            #if eta_new < 0: 
                #raise ValueError(f"Dirichelt Disnpibution parameter is positive, hoever dervired {eta_new} from orig {self._eta_} and -update {-update}")
            delta = np.abs(eta_new - self._eta_)
            #print()

            if verbose: 
                print(f"M Step: delta eta is {delta}")

            self._eta_ = eta_new 
            it += 1 
            
            if delta < threshold: 
                return 

        warnings.warn(f"Update Eta: Maximum iteration reached at step {it}")


    def fit(self, step:int = 200, threshold:float = 1e-5, verbose:bool = False, neg_delta_patience:int = 5):
        
        it = 0
        neg_delta = 0

        while it < step: 

            elbo = compute_elbo(
                self._gamma_,
                self._phi_,
                self._lambda_,
                self._alpha_,
                self._eta_,
                self.word_ct_array
            )
            print(f"{it}-> m perpexlity {np.exp(-1 * elbo/np.sum(self.word_ct_array))}")
            if it == 0: 
                print(f'npaining started -> mean ELBO at init is :{elbo/self.M}')
    

            self.e_step(step, threshold, verbose=verbose)
            self.m_step(step, threshold, verbose=verbose)

            elbo_hat = compute_elbo(
                self._gamma_,
                self._phi_,
                self._lambda_,
                self._alpha_,
                self._eta_,
                self.word_ct_array
            )
            
            delta_elbo = elbo_hat - elbo 
            #print(f"Iteration {it}, improve on ELBO is {delta_elbo}")

            if delta_elbo < 0: 
                neg_delta += 1 

            if neg_delta > neg_delta_patience: 
                warnings.warn(f"Elbo decereases for {neg_delta} times")

            if delta_elbo > 0 and delta_elbo < threshold: 
                print(f'ELBO converged at {it} -> mean ELBO:{elbo_hat/self.M}')
                return 
            
            it += 1 
    
        print(f"Maximum iteration reached at {it} -> mean ELBO: {elbo_hat/self.M}")

        return self 





            






