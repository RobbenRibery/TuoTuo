import torch as tr 
import numpy as np 
from scipy.special import gammaln, psi 

from typing import List, Dict, Union

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

    doc_term = elbo_doc_depend_part(_eta_, _gamma_, _phi_, _lambda_, w_ct)

    return corpus_term + doc_term


def eblo_corpus_part(
        _eta_: tr.Tensor,
        _lambda_: tr.Tensor, 
        _alpha_:tr.Tensor, 
        n_docs:int,
    ) -> float: 

    if _alpha_.ndim == 2: 
        K = len(_alpha_[0])
        _alpha_ = _alpha_[0]
        _eta_ = _eta_[0]
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

    assert _phi_.ndim() == 3 
    

    M = _phi_.shape[0]
    V = _lambda_.shape[1]
    K = _lambda_.shape[0]

    expec_beta_store = tr.empty((_lambda_.shape), dtype=float)
    for k in range(K): 
        expec_beta_store[k] = expec_log_dirichlet(_lambda_[k])

    term1 = 0 
    for d in range(M): 

        term1 -= log_gamma_sum_term(_gamma_[d])

        term1 += tr.dot(
            _alpha_ - _gamma_[d],
            expec_log_dirichlet(_gamma_[d])
        )

        #get number of words, as per documnet 
        Nd = _phi_[d].shape[0] 

        for n in range(Nd): 

            term1 += tr.dot(
                _phi_[d][n],
                expec_log_dirichlet(_gamma_[d])
            )

            term1 -= tr.dot(
                _phi_[d][n],
                tr.log(_phi_[d][n])
            )

        for v in V:
            term1 += w_ct[v][d] * tr.dot(
                _phi_[d][n],
                expec_log_dirichlet(expec_beta_store[:,v])
            )

    return term1.item() 


