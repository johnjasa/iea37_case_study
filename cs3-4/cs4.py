import optimization_pyopt as opt
import layout as layout
import matplotlib.pyplot as plt
from autograd import grad
# import autograd.numpy as np

file_name_turb = 'iea37-ex-opt4.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

# opt_options = {'maxiter': 5, 'disp': True, \
#                'iprint': 2, 'ftol': 1e-7}
# opt_options = {'MAXIT': 100, 'IPRINT': 0, 'ACC': 1e-7}
opt_options = {'Major iterations limit': 40}

model = layout.Layout(file_name_turb, file_name_boundary)

opt_prob = opt.Optimization(model=model, solver='SNOPT', optOptions=opt_options)

sol = opt_prob.optimize()

print(sol)

print(model.AEP_initial)
print(model.get_AEP())

model.plot_layout_opt_results(sol)
plt.show()