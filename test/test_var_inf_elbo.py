import pytest 
from pprint import pprint

import numpy as np 
from scipy.special import loggamma 
import torch as tr 

from src.utils import expec_log_dirichlet, log_gamma_sum_term, eblo_corpus_part, elbo_doc_depend_part, compute_elbo 
from test.test_func_expec_log_dirichlet import expec_log_dirichlet_mirror


def log_gamma_sum_term_mirror(x:np.ndarray,) -> float: 

    # column vector representation of hyperparameters 
    if not isinstance(x, float):
        if x.ndim == 1: 
            x = x.reshape(1,-1)

        assert x.shape[0] == 1 and x.shape[1] >= 1 
        sum_ = np.sum(x)

        return loggamma(sum_) - np.sum(loggamma(x))

    else: 
        raise ValueError(f"Input must be an array, instead obtained {x}")

@pytest.fixture
def input_1(): 
    x = [[1,1,1],[2,2,2],[3,3,3]]
    return np.array(x)

def test_log_gamma_sum_function(input_1): 

    for i in range(input_1.shape[0]): 

        x = input_1[i,:]

        print(x, x.shape, i)

        assert isinstance(log_gamma_sum_term(np.array(x)), float)
        assert isinstance(log_gamma_sum_term_mirror(x), float)

        res = log_gamma_sum_term(np.array(x))

        assert round(res,5) == round(log_gamma_sum_term_mirror(x),5)

        if i == 0: 
            assert round(res,5) == round(np.log(2),5)
        if i == 1: 
            assert round(res,5) == round(np.log(120),5)
        if i == 2: 
            assert round(res,5) == round(np.log(40320) - 3*np.log(2),5)


def var_inf_part2_corpus_level_mirror(
        _eta_: float, 
        _lambda_: np.ndarray,
        _alpha_: np.ndarray,
        num_topics: int,
        batch_size: int,
    ) -> float:

    """Compute the Components in the ELBO, which are associated with Dirichlet Prior and Multinomial Prior at the 
    Corpus level

    Returns:
        _type_: _description_
    """
    if _alpha_.ndim == 2: 
        K = len(_alpha_[0])
        _alpha_ = _alpha_[0]

    else:
        K = len(_alpha_) 

    V = _lambda_.shape[1]

    print(f"Mirror function: num_topic{K}, num_word:{V}")
    
    assert num_topics == K

    term2_1 = K*loggamma(V*_eta_) - K * V * loggamma(_eta_)

    delta = 0
    for i in range(K): 
        delta += np.sum(expec_log_dirichlet_mirror(_lambda_[i])) * (_eta_-1)
    term2_1 += delta

    term2_2 = 0
    for i in range(K): 

        delta = log_gamma_sum_term_mirror(_lambda_[i]) + \
            np.dot(
                _lambda_[i]-1, 
                expec_log_dirichlet_mirror(_lambda_[i]),
        )
        term2_2 += delta 

    term2_3 = log_gamma_sum_term_mirror(_alpha_) * batch_size
    print(f"Mirror term1 : {term2_1}")
    print(f"Mirror term2 : {term2_2}")
    print(f"Mirror term3 : {term2_3}")
    
    return term2_1 - term2_2 + term2_3

@pytest.fixture 
def input_2(): 

    # 5 words, 
    # 3 topics 
    _init_var = {
        '_eta_': 2.0,
        '_lambda_': np.array([
            [0.2, 0.2, 0.2, 0.2, 0.2],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.2, 0.3, 0.2, 0.2],
        ]),
        '_alpha_': np.array(
            [[1,2,3]],
        )
    }

    return _init_var 

def test_corpus_part_ELBO(input_2): 

    vars_np = input_2
    for k,v in vars_np.items(): 
        vars_np[k] = v

    elbo1 = eblo_corpus_part(
        vars_np['_eta_'],
        vars_np['_lambda_'],
        vars_np['_alpha_'],
        2, 
    )
    print(elbo1)

    elbo2 = var_inf_part2_corpus_level_mirror(
        vars_np['_eta_'],
        vars_np['_lambda_'],
        vars_np['_alpha_'],
        3,
        2, 
    )
    print(elbo2)
    assert elbo1 == elbo2

    
def var_inf_part1_doc_level_mirror(
    _alpha_: np.ndarray,
    _gamma_: np.ndarray,
    _phi_:np.ndarray, 
    _lambda_:np.ndarray,
    docs: np.ndarray,
): 
    if _alpha_.ndim == 2: 
        _alpha_ = _alpha_.ravel()
    
    M = _gamma_.shape[0]
    K = _gamma_.shape[1]

    print(f"Mirror Function | Number of document: {M} | Number of topics: {K}")

    term1, term2, term3, term4, term5 = 0,0,0,0,0
    for d in range(M): 

        E_gamma_d = expec_log_dirichlet_mirror(_gamma_[d])
        for i in range(K):
            term1 += (_alpha_[i]-1) * E_gamma_d[i]

            # val;idations 
            delta1 = (_alpha_[i]-1) * E_gamma_d[i]

        #get the number of words
        _phi_d =np.array(_phi_[d], dtype=float)
        Nd =  _phi_d.shape[0]
        #print(f"Mirror Function | Numnber of words {Nd}")

        term4 -= log_gamma_sum_term_mirror(_gamma_[d])
        term4 -= np.sum([(_gamma_[d][i]-1) * E_gamma_d[i] for i in range(K)])

        #val_delta = 0
        for n in range(Nd):
            #print(f"Mirror Functino, iterate: {n}") 

            wn_idx_inv = docs[d][n]

            term2 += np.sum([_phi_d[n, i] * E_gamma_d[i] for i in range(K)])

            term3 += np.sum([_phi_d[n, i] * expec_log_dirichlet_mirror(_lambda_[i])[wn_idx_inv] for i in range(K)])
            #delta2 = -log_gamma_sum_term_mirror(_gamma_[d])
            #delta3 = -np.sum([(_gamma_[d][i]-1) * E_gamma_d[i] for i in range(K)])

            term5 -= np.sum(_phi_d[n,i] * np.log(_phi_d[n,i]) for i in range(K))

        #print(term1, term4, term1+term4)

        #print(val_delta)

    print(term1+term4, term2+term5, term3)

    return term1 + term2 + term3 + term4 + term5 


@pytest.fixture
def input3_(): 

    # 3 topics 
    # 2 documents 
        ## first document has 5 words, 
        ## secondd document has 8 words 
        ## total vocab size is 10 
    # 

    _init_var = {
        '_eta_': 2,
        '_alpha_': np.array(
            [[1,2,3]],
        ),
        '_phi_': np.array(
            [
                [ # cols = topic, # column = word
                    [0.3, 0.3, 0.4],
                    [0.3, 0.3, 0.4],    
                    [0.3, 0.3, 0.4],
                    [0.3, 0.3, 0.4], 
                    [0.3, 0.3, 0.4], # document all words is at first vocab
                ],
                [
                    [0.3, 0.3, 0.4], # document words is at the first 8 vocabs
                    [0.2, 0.2, 0.8],    
                    [0.2, 0.2, 0.8],
                    [0.2, 0.2, 0.8], 
                    [0.2, 0.2, 0.8],
                    [0.2, 0.2, 0.8],
                    [0.2, 0.2, 0.8],    
                    [0.2, 0.2, 0.8],
                ],
            ], dtype=object
        ),
        '_phi_v':np.array(
            [
                [
                    [0.3,0.3,0.4],#0
                    [0.0,0.0,0.0],#1
                    [0.0,0.0,0.0],#2
                    [0.0,0.0,0.0],#3
                    [0.0,0.0,0.0],#4
                    [0.0,0.0,0.0],#5
                    [0.0,0.0,0.0],#6
                    [0.0,0.0,0.0],#7
                    [0.0,0.0,0.0],#8
                    [0.0,0.0,0.0],#9
                ],
                [
                    [0.3,0.3,0.4],#0
                    [0.2,0.2,0.8],#1
                    [0.2,0.2,0.8],#2
                    [0.2,0.2,0.8],#3
                    [0.2,0.2,0.8],#4
                    [0.2,0.2,0.8],#5
                    [0.2,0.2,0.8],#6
                    [0.2,0.2,0.8],#7
                    [0.0,0.0,0.0],#8
                    [0.0,0.0,0.0],#9
                ],
            ], dtype=float
        ),
        '_gamma_':np.array(
            [
                [1,2,3],
                [1,2,3],
            ]
        ),
        '_lambda_': np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,],
        ]),
        'docs': np.array(
            [
                [0,0,0,0,0],
                [0,1,2,3,4,5,6,7],
            ], dtype=object
        ),
        'w_ct':np.array(
            [
                [
                    5,1
                ],# word 0
                [
                    0,1
                ],# word 1
                [
                    0,1
                ], #word 2
                [
                    0,1
                ], #word 3
                [
                    0,1
                ], #word 4
                [
                    0,1
                ], #word 5
                [
                    0,1
                ], #word 6
                [
                    0,1
                ], #word 7
                [
                    0,0
                ], #word 8
                [
                    0,0
                ], #word 9
            ]
        )
    }

    # check on the input size 

    assert _init_var['_alpha_'].shape == (1,3)
    assert np.array(_init_var['_phi_'][0],dtype=float).shape == (5,3)
    assert np.array(_init_var['_phi_'][1],dtype=float).shape == (8,3)
    assert _init_var['_gamma_'].shape == (2,3)
    assert _init_var['_lambda_'].shape == (3,10)
    assert np.array(_init_var['docs'][0], dtype=float).shape == (5,)
    assert np.array(_init_var['docs'][1], dtype=float).shape == (8,)
    assert _init_var['w_ct'].shape == (10,2)

    return  _init_var


def test_docs_part_ELBO(input3_): 

    #pprint(input3_)

    doc_elbo_mirror = var_inf_part1_doc_level_mirror(
        input3_['_alpha_'],
        input3_['_gamma_'],
        input3_['_phi_'],
        input3_['_lambda_'],
        input3_['docs']
    )

    doc_elbo = elbo_doc_depend_part(
        input3_['_alpha_'],
        input3_['_gamma_'],
        input3_['_phi_v'],
        input3_['_lambda_'],
        input3_['w_ct'],
    )
    #print(doc_elbo)
    print(f"Mirror func doc elbo -> {doc_elbo_mirror}")

    assert round(doc_elbo_mirror,5) == round(doc_elbo,5)


    corpus_eblo_mirror = var_inf_part2_corpus_level_mirror(
        input3_['_eta_'],
        input3_['_lambda_'],
        input3_['_alpha_'],
        input3_['_gamma_'].shape[1],
        input3_['_gamma_'].shape[0], 
    )

    print(f"Mirror func corpus elbo -> {corpus_eblo_mirror}")

    elbo = compute_elbo(
        input3_['_gamma_'],
        input3_['_phi_v'],
        input3_['_lambda_'],
        input3_['_alpha_'],
        input3_['_eta_'],
        input3_['w_ct'],
    )

    #print(corpus_eblo_mirror)
    #print(elbo)

    elbo_mirror = doc_elbo_mirror + corpus_eblo_mirror

    assert elbo == elbo_mirror


