import torch as tr 
import numpy as np 
from scipy.special import gammaln, psi 


def exp_log_dirichlet(dirichlet_parm:tr.Tensor,) -> tr.Tensor: 

    """Compute the E[log x_i | dirichlet_parm] for every single dimension, return as a tensor 

    Returns:
        tr.Tensor: E[log x|dirichlet_parm] for every dimension of the dirichlet variable 
    """

    assert sum(dirichlet_parm > 0) == len(dirichlet_parm), f"Parameters for Dirichilet distribution should be all positive, get {dirichlet_parm}"

    term2 = tr.digamma(tr.sum(dirichlet_parm))

    temr1s = tr.diagamma(dirichlet_parm)

    assert temr1s.shape == dirichlet_parm.shape

    return temr1s - term2

def dirichlet_expectation(alpha):
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])
