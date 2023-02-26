import torch as tr 
import numpy as np 

from src.utils import expec_log_dirichlet, expec_log_dirichlet_mirror
tensor1 = tr.tensor([[1,1,1]], dtype=float)
tensor2 = tr.tensor([1,1,1], dtype=float)
tensor3 = tr.tensor([4,5,6], dtype=float)
tensor4 = tr.tensor([[0.9,1,0.2]], dtype=float)
# print(tensor.ndim)
# print(tr.sum(tensor))
for tensor in [tensor1, tensor2, tensor3, tensor4]:
    print(tensor)
    exp_dirichlet_var_ = expec_log_dirichlet(tensor)
    print(exp_dirichlet_var_)

    exp_dirichlet_var = expec_log_dirichlet_mirror(tensor.numpy())
    print(exp_dirichlet_var)
    #print(type(exp_dirichlet_var))
    #print(exp_dirichlet_var.shape)

    assert (
        np.round(exp_dirichlet_var_.numpy(),2) == np.round(exp_dirichlet_var,2)
    ).any(), \
        print(
        np.round(exp_dirichlet_var_.numpy(),2), 
        np.round(exp_dirichlet_var,2)
    )
    print()
