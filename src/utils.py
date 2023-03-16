import torch as tr 
import numpy as np 
import pandas as pd 
from scipy.special import gammaln, psi 

from src.text_pre_processor import (
    remove_accented_chars, 
    remove_special_characters, 
    remove_punctuation,
    remove_extra_whitespace_tabs,
    remove_stopwords 
)

import copy 
from typing import List, Dict, Union
from collections import defaultdict 

DTYPE = tr.double

def data_loader(dataset_name:str, ): 

    if dataset_name == 'ap': 

        file = open('../data/ap/ap.txt', 'r+',)

        docs_raw = defaultdict(str)
        index_of_doc = None 
        ave_length = 0
        for i, line in enumerate(file):   

            if '<DOCNO>' in line: 

                key = line.strip('</DOCNO>').strip('</DOCNO>\n').strip(' ')
                index_of_doc = i

            if index_of_doc is not None:
                if i == index_of_doc + 2: 

                    assert i%3 == 0 
                    
                    if key in docs_raw: 
                        print(f'Duplicated keys detected at {key}')

                    docs_raw[key] = line 
                    ave_length += len(line.split(' '))

        print(f"There are {len(docs_raw)} documents in the dataset")
        print(f"On average estimated document length is {ave_length/len(docs_raw)} words per document")

        word_2_idx = {}
        vocab_file = open('../data/ap/vocab.txt','r+')
        for i, line in enumerate(vocab_file):
            word_2_idx[line.strip('\n')] = i

        print(f"There are {len(word_2_idx)} unique vocab in the raw corpus")

        idx_2_word = {v:k for k,v in word_2_idx.items()}

        return docs_raw, word_2_idx, idx_2_word


def text_pipeline(s:str) -> str: 

    #print(len(text))
    s = remove_accented_chars(s)

    #print(len(s1))
    s = remove_special_characters(s)

    #print(len(s1))
    s = remove_punctuation(s)
    #print(len(s1))
    s = remove_extra_whitespace_tabs(s)

    s = remove_stopwords(s)
    #print(len(s1))
    return s


def process_documents(doc_dict:dict, sample:bool = True,): 

    proc_doc_dict = copy.deepcopy(doc_dict)

    length = 0
    for k, v in doc_dict.items(): 

        new_text = text_pipeline(v)
        new_text_list = new_text.split(' ')
        proc_doc_dict[k] = new_text_list if not sample else new_text_list[:10]

        length += len(proc_doc_dict[k])

    docs_list = list(proc_doc_dict.values()) if not sample else list(proc_doc_dict.values())[:300]

    print(f"There are {len(docs_list)} documents in the dataset after processing")
    print(f"On average estimated document length is {round(length/len(doc_dict),1)} words per document after processing")

    #print(len(docs_list))
    word_ct_dict = get_vocab_from_docs(docs_list)
    word_ct_dict:dict[List]
    #print(len(word_ct_dict))

    word_ct_np, word_2_idx = get_np_wct(word_ct_dict, docs_list)
    word_ct_np:np.ndarray
    word_2_idx:dict[int]

    assert len(word_ct_dict) == word_ct_np.shape[0]
    assert len(word_ct_dict) == len(word_2_idx)
    assert len(docs_list) == word_ct_np.shape[1]

    print(f"There are {len(word_2_idx)} unique vocab in the corpus after processing")

    idx_2_word = {v:k for k,v in word_2_idx.items()} 

    return {
        'documents':docs_list,
        'vocab_doc_count_dict': word_ct_dict,
        'vocab_doc_count_array': word_ct_np,
        'vocab_to_idx': word_2_idx,
        'idx_to_vocab':idx_2_word,
    }


def expec_log_dirichlet(dirichlet_parm:tr.Tensor,) -> tr.Tensor: 

    """Compute the E[log x_i | dirichlet_parm] for every single dimension, return as a tensor 

    Returns:
        tr.Tensor: E[log x|dirichlet_parm] for every dimension of the dirichlet variable 
    """
    if dirichlet_parm.ndim == 1:
        assert sum(dirichlet_parm > 0) == len(dirichlet_parm), f"Parameters for Dirichilet distribution should be all positive, get {dirichlet_parm}" 
    else: 
        assert dirichlet_parm.shape[0] == 1, f"only vector is accepted, {dirichlet_parm.shape} matrix is provided."
        assert tr.sum(dirichlet_parm > 0).item() == dirichlet_parm.shape[1]

    term2 = tr.special.digamma(tr.sum(dirichlet_parm))
    temr1s = tr.special.digamma(dirichlet_parm)

    assert temr1s.shape == dirichlet_parm.shape 
    return temr1s - term2

def log_gamma_sum_term(x:tr.Tensor) -> float:

    """Compute 

    LogGamma(SumDims) - SumDims(LogGamma(x_i)) as this appears in the elbo formula frequently 

    Returns:
        _type_: _description_
    """

    # column vector representation of hyperparameters 
    if x.ndim == 1: 
        x = x.reshape(-1,1)

    assert x.shape[0] >= 1 and x.shape[1] == 1 
    sum_ = tr.sum(x)

    return (tr.lgamma(sum_) - tr.sum(tr.lgamma(x))).item()


def compute_elbo(
        _gamma_:tr.Tensor, 
        _phi_:tr.Tensor, 
        _lambda_:tr.Tensor,
        _alpha_: tr.Tensor,
        _eta_: tr.Tensor,
        w_ct: tr.Tensor, 
    ) -> float: 

    """Compute the approximated value for the ELBO, which is the objective function in EM steps, against a batch of documents 

    Under the surogate variational distribution q ~ q (theta, z, beta| _gamma_, _phi_, _lambda_ )

        Here we copy the decoupled DGM, we have the following 

            _gamma_ (D*K, indexed by the documnet) -> theta (dirichlet parametrised by _gamma_) 
            _phi_ (D*N[varying]*K, indexed by the documment) -> z (multinomials para) 
            _lambda_ (V*K, shared across documents) -> beta (dirichlet parameterised by _lambda_)


    Under the real posterior p that we are trying to approximate during the E stps 

        we have the folloing 

            _eta_ (W, shared across docs), can be a symmetrical/exchangeable Dirichlet parametr
            _alpha_ (K, shared across docs) 

    word_cts: 
        - List containing the dictionary:
                - vocab -> count of the vocab in the d_th document 
            there are d documents in this batch that we are computing againsr 
            Those word cts are collected on top of the training corpus 
    
    word_inds: 
        - Dictionaty containing all the words (globally across all training corpus)
        - Mapping vocab v into its index in the _lambda_ and _phi_ matrix 


    NOTE: batch_size = len(word_cts) 

    Note for the batch sending in, 

    Returns:
        _type_: _description_
    """
    _gamma_.shape[0] == _phi_.shape[0]

    n_docs = _gamma_.shape[0]

    corpus_term =  eblo_corpus_part(_eta_, _lambda_, _alpha_, n_docs)
    #print(corpus_term)

    doc_term = elbo_doc_depend_part(_alpha_, _gamma_, _phi_, _lambda_, w_ct,)
    #print(doc_term)

    return corpus_term + doc_term


def eblo_corpus_part(
        _eta_: tr.Tensor,
        _lambda_: tr.Tensor, 
        _alpha_:tr.Tensor, 
        n_docs:int,
    ) -> float: 

    if _alpha_.ndim == 2: 
        K = len(_alpha_[0])
        _alpha_ = _alpha_.flatten()
        _eta_ = _eta_.flatten()
    else:
        K = len(_alpha_) 


    # part 2, the global part of the ELBO, this part of the parameters are shared across documnets 
    term2 = log_gamma_sum_term(_alpha_) * n_docs

    for k in range(K): 

        k_sum_2 = 0 
        k_sum_2 += log_gamma_sum_term(_eta_)

        #print(_eta_ - _lambda_[k,:]) 
        k_sum_2 += tr.dot((_eta_ - _lambda_[k]).flatten(), expec_log_dirichlet(_lambda_[k]))
        k_sum_2 -= log_gamma_sum_term(_lambda_[k])

        term2 += k_sum_2

    return term2.item()


def elbo_doc_depend_part(
        _alpha_: tr.Tensor, 
        _gamma_: tr.Tensor, 
        _phi_: tr.Tensor,
        _lambda_:tr.Tensor,
        w_ct:tr.Tensor, 
    ) -> float:

    #print(w_ct)
    #print(w_ct.shape)

    _alpha_ = _alpha_.double()
    _gamma_ = _gamma_.double()
    _lambda_ = _lambda_.double()
    w_ct = w_ct.double()
    
    if _alpha_.ndim == 2: 
        _alpha_ = _alpha_.flatten()

    M = _gamma_.shape[0]
    V = _lambda_.shape[1]
    K = _gamma_.shape[1]

    #print(f"Optimised Function | Number of document: {M} | Number of topics: {K}")

    expec_beta_store = tr.empty((_lambda_.shape), dtype=float)
    for k in range(K): 
        expec_beta_store[k] = expec_log_dirichlet(_lambda_[k])

    term1, term2, term3 = 0, 0,0
    for d in range(M): 

        term1 -= log_gamma_sum_term(_gamma_[d])

        expect_theta_d = expec_log_dirichlet(_gamma_[d])
        term1 += tr.dot(
            _alpha_ - _gamma_[d],
            expect_theta_d,
        ).item()

        for v in range(V): 

            if w_ct[v][d] == 0: 
                assert _phi_[d][v].sum() == 0
                continue

            term2 += (w_ct[v][d]*tr.dot(_phi_[d][v], expect_theta_d)).item()

            if _phi_[d][v].sum() == 0: 
                logs = tr.zeros(_phi_[d][v].shape, dtype=DTYPE)
            else:
                logs = tr.log(_phi_[d][v])

            
            term2 -= (w_ct[v][d]*tr.dot(_phi_[d][v], logs)).item()
            
            term3 += (w_ct[v][d]*tr.dot(_phi_[d][v], expec_beta_store[:,v])).item()

    #print(term1, term2, term3)
    return term1 + term2 + term3


def get_vocab_from_docs(docs:List[List[str]]): 

    w_ct_dict = {}

    for id, doc in enumerate(docs):

        for word in doc: 

            if word not in w_ct_dict: 
                w_ct_dict[word] = [0]*len(docs)
            
            w_ct_dict[word][id] += 1 

    return w_ct_dict

def get_np_wct(w_ct_dict:Dict, docs:List[List[str]]):

    w_ct_np = np.empty((len(w_ct_dict), len(docs)), dtype=float)

    w_2_idx = {}

    w_idx = 0 
    for w, cts in w_ct_dict.items(): 

        w_ct_np[w_idx] = cts 
        w_2_idx[w] = w_idx 

        w_idx += 1 

    return w_ct_np, w_2_idx 


def np_obj_2_tr(x:np.ndarray) -> tr.Tensor: 

    o = tr.from_numpy(np.array(x, dtype=float))
    o = o.double()

    return o