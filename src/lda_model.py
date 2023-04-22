from typing import List, Dict
import warnings 

import torch as np 
import numpy as np 
from scipy.special import psi, gammaln

from src.utils import (
    get_vocab_from_docs, 
    get_np_wct, 
    expec_log_dirichlet,
    log_gamma_sum_term, 
    compute_elbo,
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
) -> int: 
    
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
            word_ct_array: np.ndarray = None,
            word_to_idx: dict = None, 
            idx_to_word: dict = None, 
            verbose: bool = True,
        ) -> None:

        assert get_vocab_from_docs(docs) == word_ct_dict

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
        self._alpha_ = np.ones((1,self.K))#np.random.gamma(shape = 100, scale = 0.01, size =self.K)
        self._alpha_ = self._alpha_.ravel()

        if verbose:
            print(f"Topic Dirichlet Prior, Alpha")
            print(self._alpha_.shape)
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

        #Multinomial Prior, Surrogate for Theta vector drawn from Dirichlet(Alpha)
        phi = np.zeros((len(docs),self.word_ct_array.shape[0],num_topics), dtype=DTYPE)

        print('loop phi')
        for id, d in enumerate(docs): 
            for word in d: 

                v = self.word_2_idx[word]
                phi[id][v] = 1/self.K
        
        print('looped')
        self._phi_ = phi
        print('double')
        if verbose: 
            print(f"Var -Inf - Word wise Topic Multinomial/Categorical, Phi")
            print(self._phi_.shape)


    def e_step(self, step:int = 100, threshold:float = 1e-07, verbose:bool = False,) -> None: 

        delta_gamma =  np.full(self._gamma_.shape, fill_value=np.inf)
        l2_delta_gamma = np.linalg.norm(delta_gamma)

        it = 0 
        while it < step:

            if verbose: 
                elbo = compute_elbo(
                    self._gamma_,
                    self._phi_,
                    self._lambda_,
                    self._alpha_,
                    self._eta_,
                    self.word_ct_array
                )
                print(f'Iteration {it}, Delta Gamma = {l2_delta_gamma}, the ELBO is {elbo}')
    
            gamma = np.copy(self._gamma_)
            
            ### Update Phi[d][v][k] ###
            for d in range(self.M): 
                #print(d)
                EqThetaD = expec_log_dirichlet(self._gamma_[d])
                for v in range(self.V): 

                    #if word v is not in document d, we continue to the next one 
                    if self._phi_[d][v].sum() == 0:
                        continue
                    elif round(self._phi_[d][v].sum().item()) == 1:
                        for k in range(self.K):

                            EqBetak = expec_log_dirichlet(self._lambda_[k])

                            self._phi_[d][v][k] = np.exp(EqThetaD[k] + EqBetak[v])
                    else:
                        raise ValueError(f"Sum of the multinomial parameters at document {d} and word {v} not eual to one, instead found {self._phi_[d][v]}")

                    ## -- normalisation -- ## 
                    self._phi_[d][v] /= np.sum(self._phi_[d][v])
                    if np.isnan(self._phi_[d][v]).any():
                        raise ValueError("phi nan")
                    #print(self.idx_2_word[v])
                    #print(self._phi_[d][v])
                    #print(f"{d} -> {self.idx_2_word[v]} -> {self._phi_[d][v]}")

            ### Update Lambda[k][v] ###
            for k in range(self.K):
                for v in range(self.V):
                    self._lambda_[k][v] = self._eta_ + np.dot(self.word_ct_array[v], self._phi_[:,v,k])    
                    if self._lambda_[k][v] < 0: 
                        raise ValueError(f"Varitional Dirichlet parameter at doc {d}, topic {k} went negative after update using alpha:{self.self._eta_} and phi part {np.dot(self.word_ct_array[v], self._phi_[:,v,k])}")
            
            ### Update Gamma[d][k] ###
            for d in range(self.M): 
                for k in range(self.K):
                    gamma[d][k] = self._alpha_[k] + np.dot(self.word_ct_array[:,d],self._phi_[d,:,k])
                    if gamma[d][k] < 0: 
                        raise ValueError(f"Varitional Dirichlet parameter at doc {d}, topic {k} went negative after update using alpha:{self._alpha_[k]} and phi part {np.dot(self.word_ct_array[:,d],self._phi_[d,:,k])}")
                    ###print(self._gamma_[d][k], gamma[d][k])


            delta_gamma = self._gamma_ - gamma
            l2_delta_gamma = np.linalg.norm(delta_gamma)

            if l2_delta_gamma < threshold: 
                return 

            self._gamma_ = gamma
            it += 1 

        warnings.warn(f"Update phi, lambda, gamma: Maximum iteration reached at step {it}")
    

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





            






