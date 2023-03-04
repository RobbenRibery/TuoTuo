import pytest 
import numpy as np 
from scipy.special import loggamma 
import torch as tr 

from src.utils import expec_log_dirichlet, log_gamma_sum_term
from test.test_func_expec_log_dirichlet import expec_log_dirichlet_mirror

def log_gamma_sum_term(x:tr.Tensor) -> float: 

    # column vector representation of hyperparameters 
    if x.ndim == 1: 
        x = x.reshape(1,-1)

    assert x.shape[0] == 1 and x.shape[1] >= 1 
    sum_ = tr.sum(x)

    return (tr.lgamma(sum_) - tr.sum(tr.lgamma(x))).item()

def log_gamma_sum_term_mirror(x:np.ndarray,) -> float: 

    # column vector representation of hyperparameters 
    if x.ndim == 1: 
        x = x.reshape(1,-1)

    assert x.shape[0] == 1 and x.shape[1] >= 1 
    sum_ = np.sum(x)

    return loggamma(sum_) - np.sum(loggamma(x))

@pytest.fixture
def input_1(): 
    x = [[1,1,1],[2,2,2],[3,3,3]]
    return np.array(x)

def test_log_gamma_sum_function(input_1): 

    for i in range(input_1.shape[0]): 

        x = input_1[i,:]

        print(x, x.shape, i)

        assert isinstance(log_gamma_sum_term(tr.tensor(x)), float)
        assert isinstance(log_gamma_sum_term_mirror(x), float)

        res = log_gamma_sum_term(tr.tensor(x))

        assert round(res,5) == round(log_gamma_sum_term_mirror(x),5)

        if i == 0: 
            assert round(res,5) == round(np.log(2),5)
        if i == 1: 
            assert round(res,5) == round(np.log(120),5)
        if i == 2: 
            assert round(res,5) == round(np.log(40320) - 3*np.log(2),5)


def var_inf_part2_corpus_level(
        _eta_: tr.Tensor,
        _lambda_: tr.Tensor, 
        _alpha_:tr.Tensor, 
        num_topics:int, 
        batch_size:int,
    ) -> float: 

    if _alpha_.ndim == 2: 
        K = len(_alpha_[0])
        _alpha_ = _alpha_[0]
        _eta_ = _eta_[0]
    else:
        K = len(_alpha_) 

    assert num_topics == K

    # part 2, the global part of the ELBO, this part of the parameters are shared across documnets 
    term2 = log_gamma_sum_term(_alpha_) * batch_size

    for k in range(K): 

        k_sum_2 = 0 
        k_sum_2 += log_gamma_sum_term(_eta_)

        #print(_eta_ - _lambda_[k,:]) 
        k_sum_2 += tr.dot((_eta_ - _lambda_[k]).flatten(), expec_log_dirichlet(_lambda_[k]))
        k_sum_2 -= log_gamma_sum_term(_lambda_[k])

        term2 += k_sum_2

    return term2.item()


def var_inf_part2_corpus_level_mirror(
        _eta_: np.ndarray, 
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
        _eta_ = _eta_[0]
    else:
        K = len(_alpha_) 
    
    assert num_topics == K

    term2_1 = K*log_gamma_sum_term_mirror(_eta_) 
    for i in range(K): 

        delta = np.dot(_eta_-1, expec_log_dirichlet_mirror(_lambda_[i])) 

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

    
    return term2_1 - term2_2 + term2_3

@pytest.fixture 
def input_2(): 

    # 5 words, 
    # 3 topics 

    _init_var = {
        '_eta_': np.array(
            [[2,2,2,2,2]],
        ),
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

    vars_tr = vars_np.copy()
    for k,v in vars_tr.items(): 
        vars_tr[k] = tr.from_numpy(v)

    elbo1 = var_inf_part2_corpus_level(
        vars_tr['_eta_'],
        vars_tr['_lambda_'],
        vars_tr['_alpha_'],
        3,
        2, 
    )

    elbo2 = var_inf_part2_corpus_level_mirror(
        vars_np['_eta_'],
        vars_np['_lambda_'],
        vars_np['_alpha_'],
        3,
        2, 
    )

    assert round(elbo1,5) == round(elbo2,5)

    



