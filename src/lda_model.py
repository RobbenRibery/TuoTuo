from typing import List, Dict

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

    """Class implemented the 

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

        if verbose: 
            print(f"Word mapping to vocab index is: ")
            pprint(self.word_2_idx)

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
            print(self._alpha_)
            print() 
        # Dirichlet Prior - Exchangeable Dirichlet
        self._eta_ = tr.ones(1,self.V, dtype=DTYPE)
        self._eta_ = self._eta_.flatten()

        if verbose:
            print(f"Word Dirichlet Prior, Eta")
            print(self._eta_.shape)
            print(self._eta_)
            print()


        # define the Convexity-based Varitional Inference hyper-parameters 
        #Dirichlet Prior, Surrogate for _eta_ 
        self._lambda_ = tr.ones(self.K, self.V, dtype=DTYPE)
        if verbose: 
            print(f"Var Inf - Word Dirichlet prior, Lambda")
            print(self._lambda_.shape)
            print(self._lambda_)
            print()

        #Dirichlet Prior, Surrogate for _alpha_ 
        self._gamma_ = self._alpha_ + self.V/self.K
        self._gamma_ = self._gamma_.expand(self.M,-1)

        if verbose: 
            print(f"Var Inf - Topic Dirichlet prior, Gamma")
            print(self._gamma_.shape)
            print(self._gamma_)
            print()

        #Multinomial Prior, Surrogate for Theta vector drawn from Dirichlet(Alpha)
        phi = tr.zeros(len(docs),self.word_ct_array.shape[0],num_topics)

        for id, d in enumerate(docs): 
            for word in d: 

                v = self.word_2_idx[word]
                phi[id][v] = 1/self.K

        self._phi_ = phi.double()

        if verbose: 
            print(f"Var -Inf - Word wise Topic Multinomial/Categorical, Phi")
            print(self._phi_.shape)
            print(self._phi_)


    def e_step(self, threshold:float = 1e-08, verbose:bool = True,) -> None: 

        delta_gamma =  tr.full(self._gamma_.shape, fill_value=tr.inf)
        l2_delta_gamma = tr.norm(delta_gamma)

        i = 0 
        while l2_delta_gamma > threshold:

            if verbose: 
                i+= 1 
                print(f'Iteration {i}, Delta Gamma = {l2_delta_gamma.item()}')

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

            self._gamma_ = gamma
    

    def m_step(self,): 


        return None 
