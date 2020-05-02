import optimization as opt
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as np

file_name_turb = 'iea37-ex-opt3.yaml'
file_name_boundary = 'iea37-boundary-cs3.yaml'

opt_options = {'maxiter': 50, 'disp': True, \
               'iprint': 2, 'ftol': 1e-10}

cs3 = opt.cs3Opt(file_name_turb, file_name_boundary, opt_options=opt_options, jac=False)
grad_func = grad(cs3._get_AEP_opt)

cs3.set_grad(grad_func)

results = cs3.om_optimize()

print(results)

cs3.plot_layout_opt_results()

plt.show()



