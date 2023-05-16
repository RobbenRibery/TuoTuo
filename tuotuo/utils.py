import torch as tr 
import numpy as np 
import pandas as pd 
from scipy.special import gammaln, psi

from tuotuo.text_pre_processor import (
    remove_accented_chars, 
    remove_special_characters, 
    remove_punctuation,
    remove_extra_whitespace_tabs,
    remove_stopwords 
)
import warnings 
import copy 
from typing import List, Dict, Union
from collections import defaultdict 

DTYPE = float 

def np_clip_for_exp(x:np.ndarray, sub:float = 1e-05): 

    if x >0: 
        x = max(sub, x)
    elif x< 0: 
        x = min(sub,x)
    return x


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


def expec_log_dirichlet(dirichlet_parm:np.ndarray,) -> np.ndarray: 

    """Compute the E[log var | dirichlet_parm] for every single dimension, return as a tensor 

    Returns:
        np.ndarray: E[log var|dirichlet_parm] for every dimension of the dirichlet variable 
    """
    if dirichlet_parm.ndim == 1:
        assert sum(dirichlet_parm > 0) == len(dirichlet_parm), f"Parameters for Dirichilet distribution should be all positive, get {dirichlet_parm}" 

        term2 = psi(np.sum(dirichlet_parm))
        temr1s = psi(dirichlet_parm)

        assert temr1s.shape == dirichlet_parm.shape 
        return temr1s - term2
    else: 
        warnings.warn('Please use Cython function for 2d Dirichlet Expectation')
        assert np.sum(dirichlet_parm > 0) == dirichlet_parm.shape[1]*dirichlet_parm.shape[0]

        return psi(dirichlet_parm) - psi(np.sum(dirichlet_parm, 1))[:, np.newaxis]

def log_gamma_sum_term(x:np.ndarray) -> float:

    """Compute 

    LogGamma(SumDims) - SumDims(LogGamma(x_i)) as this appears in the elbo formula frequently 

    Returns:
        _type_: _description_
    """

    # column vector representation of hyperparameters 
    if x.ndim == 1: 
        x = x.reshape(-1,1)

    assert x.shape[0] >= 1 and x.shape[1] == 1 
    sum_ = np.sum(x)

    assert sum_ >= 0 
    assert (x >= 0).all()

    return gammaln(sum_) - np.sum(gammaln(x))


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
