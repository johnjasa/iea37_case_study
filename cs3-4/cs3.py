import optimization as opt
import matplotlib.pyplot as plt
from autograd import grad

file_name_turb = 'iea37-ex-opt3.yaml'
file_name_boundary = 'iea37-boundary-cs3.yaml'

opt_options = {'maxiter': 20, 'disp': True, \
               'iprint': 2, 'ftol': 1e-7}

cs3 = opt.cs3Opt(file_name_turb, file_name_boundary, opt_options=opt_options, jac=True)

cs3.optimize()

cs3.plot_layout_opt_results()

plt.show()



