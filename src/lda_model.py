from typing import List, Dict
import warnings 

import torch as tr 
import numpy as np 

from src.utils import (
    get_vocab_from_docs, 
    get_np_wct, 
    np_obj_2_tr,
    expec_log_dirichlet,
    log_gamma_sum_term, 
    compute_elbo,
)

from pprint import pprint

SEED = 42
DTYPE = tr.double 

class LDASmoothed: 

    """Class implemented the Smoothed Version of Latent Dirichlet Allocation

    - Parameters init 
    - Training 
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

        self.word_ct_array = tr.from_numpy(word_ct_array).double()

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
        self._alpha_ = np.random.gamma(100, 0.01, (1,self.K))
        self._alpha_ = tr.from_numpy(self._alpha_)
        self._alpha_ = self._alpha_.double()
        self._alpha_ = self._alpha_.flatten()

        if verbose:
            print(f"Topic Dirichlet Prior, Alpha")
            print(self._alpha_.shape)
            print() 
        # Dirichlet Prior - Exchangeable Dirichlet
        self._eta_ = tr.ones(1,self.V, dtype=DTYPE)
        self._eta_ = self._eta_.flatten()

        if verbose:
            print(f"Word Dirichlet Prior, Eta")
            print(self._eta_.shape)
            print()


        # define the Convexity-based Varitional Inference hyper-parameters 
        #Dirichlet Prior, Surrogate for _eta_ 
        self._lambda_ = tr.ones(self.K, self.V, dtype=DTYPE)
        if verbose: 
            print(f"Var Inf - Word Dirichlet prior, Lambda")
            print(self._lambda_.shape)
            print()

        #Dirichlet Prior, Surrogate for _alpha_ 
        self._gamma_ = self._alpha_ + self.V/self.K
        self._gamma_ = self._gamma_.expand(self.M,-1)

        if verbose: 
            print(f"Var Inf - Topic Dirichlet prior, Gamma")
            print(self._gamma_.shape)
            print()

        #Multinomial Prior, Surrogate for Theta vector drawn from Dirichlet(Alpha)
        phi = tr.zeros(len(docs),self.word_ct_array.shape[0],num_topics)

        print('loop phi')
        for id, d in enumerate(docs): 
            for word in d: 

                v = self.word_2_idx[word]
                phi[id][v] = 1/self.K
        
        print('looped')
        self._phi_ = phi.double()
        print('double')
        if verbose: 
            print(f"Var -Inf - Word wise Topic Multinomial/Categorical, Phi")
            print(self._phi_.shape)


    def e_step(self, step:int = 100, threshold:float = 1e-07, verbose:bool = True,) -> None: 

        delta_gamma =  tr.full(self._gamma_.shape, fill_value=tr.inf)
        l2_delta_gamma = tr.norm(delta_gamma)

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
                print(f'Iteration {it}, Delta Gamma = {l2_delta_gamma.item()}, the ELBO is {elbo}')
    
            gamma = self._gamma_.clone()
            
            ### Update Phi[d][v][k] ###
            for d in range(self.M): 
                EqThetaD = expec_log_dirichlet(self._gamma_[d])
                for v in range(self.V): 

                    #if word v is not in document d, we continue to the next one 
                    if self._phi_[d][v].sum() == 0:
                        continue
                    elif round(self._phi_[d][v].sum().item()) == 1:
                        for k in range(self.K):

                            EqBetak = expec_log_dirichlet(self._lambda_[k])

                            self._phi_[d][v][k] = tr.exp(EqThetaD[k] + EqBetak[v])
                    else:
                        raise ValueError(f"Sum of the multinomial parameters at document {d} and word {v} not eual to one, instead found {self._phi_[d][v]}")

                    ## -- normalisation -- ## 
                    self._phi_[d][v] = self._phi_[d][v]/tr.sum(self._phi_[d][v])
                    #print(f"{d} -> {self.idx_2_word[v]} -> {self._phi_[d][v]}")

            ### Update Lambda[k][v] ###
            for k in range(self.K):
                for v in range(self.V):
                    self._lambda_[k][v] = self._eta_[v] + tr.dot(self.word_ct_array[v], self._phi_[:,v,k])    
            
            ### Update Gamma[d][k] ###
            for d in range(self.M): 
                for k in range(self.K):
                    gamma[d][k] = self._alpha_[k] + tr.dot(self.word_ct_array[:,d],self._phi_[d,:,k])


            delta_gamma = self._gamma_ - gamma
            l2_delta_gamma = tr.norm(delta_gamma)

            if l2_delta_gamma < threshold: 
                return 

            self._gamma_ = gamma
            it += 1 

        warnings.warn(f"Update phi, lambda, gamma: Maximum iteration reached at step {it}")
    

    def m_step(self, step: int = 100, threshold:float = 1e-07, verbose:bool = True,) -> None: 

        self.update_alpha(step, threshold, verbose)
        self.update_eta(step, threshold, verbose)


    def update_alpha(self, step:int = 500, threshold:float = 1e-07, verbose:bool = True,) -> None: 

        """Newton-Raphson in Linear Time for the sepcial Hessian with 
        Diag(h) + 1z1T
        """

        it = 0 

        while it <= step: 

            sum_ = tr.sum(self._alpha_)

            # grad in R 1*K
            g = self.M * (tr.digamma(sum_)-tr.digamma(self._alpha_)) + \
                tr.sum(tr.digamma(self._gamma_), dim=0) - \
                tr.sum(tr.digamma(tr.sum(self._gamma_, dim=1)))
            
            # hessian diagonal vector in R 1*K 
            h = - self.M * tr.polygamma(1, self._alpha_)

            # hessian constant part 
            z = self.M * tr.polygamma(1, tr.sum(self._alpha_))

            # offset c 
            c = tr.sum(g/h) / ((1/z)+tr.sum(1/h))

            # newton step s
            update = (g-c)/h 

            alpha_new = self._alpha_ - update 

            delta = tr.norm(alpha_new-self._alpha_) 

            if verbose: 
                elbo = compute_elbo(
                    self._gamma_,
                    self._phi_,
                    self._lambda_,
                    self._alpha_,
                    self._eta_,
                    self.word_ct_array
                )
                print(f"Iteration {it}, Delta Alpha = {delta} elbo is {elbo}")

            self._alpha_ = alpha_new
            it += 1 

            if delta < threshold: 
                return 
            
        warnings.warn(f"Update alphda Maximum iteration reached at step {it}")

    def update_eta(self, step:int = 100, threshold:float = 1e-07, verbose:bool = True) -> None:


        """Newton-Raphson in Linear Time for the sepcial Hessian with 
        Diag(h) + 1z1T
        """
        
        it = 0 
        while it <= step: 
            
            # gradient 
            g = self.K * (tr.digamma(tr.sum(self._eta_)) - tr.digamma(self._eta_)) + \
                tr.sum(tr.digamma(self._lambda_), dim=0) - tr.sum(tr.digamma(tr.sum(self._lambda_, dim=1)))

            # h hessian diagonal 
            h = - self.K * tr.polygamma(1, self._eta_)

            # hessain constant part 
            z = self.K * tr.polygamma(1, tr.sum(self._eta_))

            # offet c 
            c = tr.sum(g/h) / ((1/z)+tr.sum(1/h))

            # newton step 
            update = (g-c)/h  

            eta_new = self._eta_ - update 

            delta = tr.norm(eta_new - self._eta_) 

            if verbose: 
                elbo = compute_elbo(
                    self._gamma_,
                    self._phi_,
                    self._lambda_,
                    self._alpha_,
                    self._eta_,
                    self.word_ct_array
                )
                print(f"Iteration {it}, Delta Alpha = {delta}, elbo is {elbo}")

            self._eta_ = eta_new 
            it += 1 
            
            if delta < threshold: 
                return 

        warnings.warn(f"Update Eta: Maximum iteration reached at step {it}")

    def fit(self, step:int = 100, threshold:float = 1e-5, verbose:bool = False, neg_delta_patience:int = 5):
        
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
            print(f"{it}->ELBO {elbo}")
            if it == 0: 
                print(f'Training started -> ELBO at init is :{elbo}')

            self.e_step(step, threshold, verbose=True)
            self.m_step(step, threshold, verbose=True)

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
                print(f'ELBO converged at {it} -> ELBO:{elbo_hat}')
                return 
            
            it += 1 
    
        warnings.warn(f"Maximum iteration reached at {it} -> ELBO: {elbo_hat}")

        return self 
 

    def predict(self,): 

        return None





            






