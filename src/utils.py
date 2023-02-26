import torch as tr 
import numpy as np 
from scipy.special import gammaln, psi 

from typing import List, Dict

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
    

def expec_log_dirichlet_mirror(alpha:np.ndarray) -> np.ndarray:
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


def compute_elbo(
        batch_size: int, 
        _gamma_:tr.Tensor, 
        _phi_:tr.Tensor, 
        _lambda_:tr.Tensor,
        _alpha_: tr.Tensor,
        _eta_: tr.Tensor,
        word_cts: List[Dict[str:int]], 
        word_inds: Dict[str:int], 
        train_mode = True, 
    ): 

    """Compute the approximated value for the ELBO, which is the objective function in EM steps, against a batch of documents 

    Under the surogate variational distribution q ~ q (theta, z, beta| _gamma_, _phi_, _lambda_ )

        Here we copy the decoupled DGM, we have the following 

            - _gamma_ (D*k, indexed by the documnet) -> theta (dirichlet parametrised by _gamma_) 
            - _phi_ (D*W*K, indexed by the documment) -> z (multinomials para) 
            - _lambda_ (W*K, shared across documents) -> beta (dirichlet parameterised by _lambda_)


    Under the real posterior p that we are trying to approximate during the E stps 

        we have the folloing 

            - _eta_ (W, shared across docs), can be a symmetrical/exchangeable Dirichlet parametr
            - _alpha_ (K, shared across docs) 

    word_cts: 
        - List containing the dictionary:
                - vocab -> count of the vocab in the d_th document 
            there are d documents in this batch that we are computing againsr 
            Those word cts are collected on top of the training corpus 
    
    # vocabs
    #     - Nested List containing the vocabs from the actual document that we are passing in: 

    #         if it's for training, this set of vocabs would be exactly the same as the list of keys in word_cts, 

    #         if it's purefly for inference purpose, than the vocbas in this list could be out of the bound 


    NOTE: batch_size = len(word_cts) 

    Note for the batch sending in, 

    Returns:
        _type_: _description_
    """
    assert batch_size == len(word_cts)

    num_topics = _alpha_.shape[1] if _alpha_.ndim == 2 else len(_alpha_)
    vocabs = list(word_inds.keys())

    # iterate through topics to collect variables 
    Eq_log_betas = {}
    for k in num_topics: 
        Eq_log_betas[k]= expec_log_dirichlet(_lambda_[k])

    # part 1, the local part of the ELBO, this part of the parameters are optimised locally/ against each document 
    term1 = 0 
    for d in range(batch_size): 

        doc_vocab = list(word_cts[d].keys())
        Eq_log_thetas = expec_log_dirichlet(_gamma_[d,:])

        v_sum = 0
        for v in vocabs:
            v:str

            # trianing mode, every word for sure is in the corpus
            count_dv = word_cts[d][v]

            # get the word index
            v_indx = word_inds[v]

            #Eq(log Theta_d, across all k dimensions), using the gamma variational distribution

            k_sum = 0 
            for k in range(num_topics):

                k_sum += (Eq_log_thetas[k] + Eq_log_betas[k][v_indx] - tr.log(_phi_[d][v_indx][k])) * _phi_[d][v_indx][k]

            v_sum += count_dv * k_sum

        term1 += v_sum 

        term1 += tr.dot((_alpha_ - _gamma_[d]), Eq_log_thetas) 

        term1 += tr.lgamma(
            tr.sum(_gamma_[d])
        )

        term1 -= tr.sum(tr.lgamma(_gamma_[d]))

    # part 2, the global part of the ELBO, this part of the parameters are shared across documnets 
    term2 = (tr.lgamma(tr.sum(_alpha_)) - tr.sum(tr.lgamma(_alpha_))) * batch_size

    for k in range(num_topics): 

        k_sum_2 = 0 

        k_sum_2 += tr.lgamma(tr.sum(_eta_)) - tr.sum(tr.lgamma(_eta_))

        k_sum_2 += tr.dot((_eta_ - _lambda_[k]), expec_log_dirichlet(_lambda_[k]))

        k_sum_2 -= tr.lgamma(tr.sum(_lambda_[k]))

        k_sum_2 += tr.sum(tr.lgamma(_lambda_[k]))

        term2 += k_sum_2/batch_size

    return term1 + term2 




    





             
                                                            











            










    return None 

