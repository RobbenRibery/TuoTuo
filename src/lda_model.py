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

        self.idx_2_word = {v:k for k,v in self.word_2_idx.items()}
        self.idx_2_word: dict

        # gather the docs in it's vocab index form 
        self.docs_v_idx = self.docs.copy()
        for id, d in enumerate(docs): 
            for n in range(len(d)): 
                self.docs_v_idx[id][n] = self.docs[docs[id][n]]

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

        if verbose:
            print(f"Topic Dirichlet Prior, Alpha")
            print(self._alpha_.shape)
            print(self._alpha_)
            print() 
        # Dirichlet Prior - Exchangeable Dirichlet
        self._eta_ = tr.ones(1,self.V, dtype=DTYPE)

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
        _phi_ = []
        for d in range(self.M): 
            _phi_.append(np.ones((len(docs[d]),self.K)) * (1/self. K))

        self._phi_ = np.array(_phi_, dtype=object)

        if verbose: 
            print(f"Var -Inf - Word wise Topic Multinomial/Categorical, Phi")
            print(self._phi_.shape)
            print(_phi_)


    def e_step(self, threshold:float = 1e-08, verbose:bool = True,) -> None: 

        delta_gamma =  tr.full(self._gamma_.shape, fill_value=tr.inf)
        l2_delta_gamma = tr.norm(delta_gamma)

        i = 0 
        while l2_delta_gamma > threshold:

            if verbose: 
                i+= 1 
                print(f'Iteration {i}, Delta Gamma = {l2_delta_gamma.item()}')

            gamma = self._gamma_.clone()
            
            # Update Phi
            for d in range(self.M): 

                phi_d = np_obj_2_tr(self._phi_[d])
                Nd = phi_d.shape[0]

                EqThetaD = expec_log_dirichlet(self._gamma_[d])

                for n in range(Nd): 

                    vocab_idx = self.docs_v_idx[d][n]

                    for k in range(self.K):

                        EqBetak = expec_log_dirichlet(self._lambda_[k])

                        phi_d[n][k] = tr.exp(EqThetaD[k] + EqBetak[vocab_idx])
                        self._phi_[d][n][k] = phi_d[n][k].numpy()

                # normalisation 
                self._phi_[d][n] = self._phi_[d][n]/np.sum(self._phi_[d][n])

            # Update Lambda 
            for v in range(self.V): 

                ###
                # loop through all document 
                
                # -> array: given v and k , find the 
                # position of word v in document d, if not appeared, return 

                ###

                self._lambda_[k][v] += tr.dot(
                    self.word_ct_array[v], 
                    tr.from_numpy(self._phi_[:][:][k]).double()
                )    

            gamma[d] = self._alpha_ + self.word_ct_array[:,d]

            delta_gamma = self._gamma_ - gamma
            l2_delta_gamma = tr.norm(delta_gamma)

            self._gamma_ = gamma
    

    def m_step(self,): 


        return None 
