import pytest 
from scipy.special import gammaln, psi 

import torch as tr 
import numpy as np 

from src.utils import expec_log_dirichlet

def expec_log_dirichlet_mirror(alpha:np.ndarray) -> np.ndarray:
    """
    For a vector theta ~ Dir(alpha), computes E[log(theta)] given alpha.
    """
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])


@pytest.fixture
def input_values():
    
    tensor1 = tr.tensor([[1,1,1]], dtype=float)
    tensor2 = tr.tensor([1,1,1], dtype=float)
    tensor3 = tr.tensor([4,5,6], dtype=float)
    tensor4 = tr.tensor([[0.9,1,0.2]], dtype=float)

    return [tensor1, tensor2, tensor3, tensor4]
    
def test_expectation_log_dirichlet(input_values):

    for tensor in input_values:
        exp_dirichlet_var_ = expec_log_dirichlet(tensor)
        #print(tensor)
        #print(exp_dirichlet_var_)

        exp_dirichlet_var = expec_log_dirichlet_mirror(tensor.numpy())
        #print(type(exp_dirichlet_var))
        #print(exp_dirichlet_var)
        #print(exp_dirichlet_var.shape)

        assert (
            np.round(exp_dirichlet_var_.numpy(),2) == np.round(exp_dirichlet_var,2)
        ).any(), \
            print(
            np.round(exp_dirichlet_var_.numpy(),2), 
            np.round(exp_dirichlet_var,2)
        )
        #print()
