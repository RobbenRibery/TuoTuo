from typing import List, Dict

import torch as tr 
import numpy as np 

from src.utils import get_vocab_from_docs, get_np_wct

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
        
        # number of documents 
        M = len(docs)

        # number of unique words in corpus 
        if word_ct_array is not None:
            assert word_ct_array.shape[0] == len(word_ct_dict)

            val_word_ct_array, val_word2idx = get_np_wct(word_ct_dict, docs)
            assert (word_ct_array == val_word_ct_array).all()

            word_2_idx = val_word2idx 
        else:
            word_ct_array, word_2_idx = get_np_wct(word_ct_dict, docs)

        self.word_2_idx = word_2_idx 
        self.idx_2_word = {v:k for k,v in self.word_2_idx.items()}

        # number of unique words in the corpus 
        V = word_ct_array.shape[0]  

        # number of topics 
        K = num_topics

        #parameters init
        # define the DGM hyper-parameters
        # Dirichlet Prior 
        self._alpha_ = np.random.gamma(100, 0.01, (1,K))
        self._alpha_ = tr.from_numpy(self._alpha_)
        self._alpha_ = self._alpha_.double()

        if verbose:
            print(f"Topic Dirichlet Prior, Alpha")
            print(self._alpha_.shape)
            print(self._alpha_)
            print() 
        # Dirichlet Prior - Exchangeable Dirichlet
        self._eta_ = tr.ones(1,V, dtype=DTYPE)

        if verbose:
            print(f"Word Dirichlet Prior, Eta")
            print(self._eta_.shape)
            print(self._eta_)
            print()


        # define the Convexity-based Varitional Inference hyper-parameters 
        #Dirichlet Prior, Surrogate for _eta_ 
        self._lambda_ = tr.ones(K, V, dtype=DTYPE)
        if verbose: 
            print(f"Var Inf - Word Dirichlet prior, Lambda")
            print(self._lambda_.shape)
            print(self._lambda_)
            print()

        #Dirichlet Prior, Surrogate for _alpha_ 
        self._gamma_ = self._alpha_ + V/K
        self._gamma_ = self._gamma_.expand(M,-1)

        if verbose: 
            print(f"Var Inf - Topic Dirichlet prior, Gamma")
            print(self._gamma_.shape)
            print(self._gamma_)
            print()

        #Multinomial Prior, Surrogate for Theta vector drawn from Dirichlet(Alpha)
        _phi_ = []
        for d in range(M): 
            _phi_.append(np.ones((len(docs[d]),K)) * (1/K))

        self._phi_ = np.array(_phi_, dtype=object)

        if verbose: 
            print(f"Var -Inf - Word wise Topic Multinomial/Categorical, Phi")
            print(self._phi_.shape)
            print(_phi_)
