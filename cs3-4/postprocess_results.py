import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


out_dir = 'two_boxes_GA_results_along'
tol = 1e-6

# results_dict = {
#     'optTime' : sol.optTime,
#     'optInform' : sol.optInform,
#     'AEP_initial' : model.AEP_initial,
#     'AEP_final' : AEP_final,
#     'locsx' : locsx,
#     'locsy' : locsy,
#     'boundary_con' : cons['boundary_con'],
#     'spacing_con' : cons['spacing_con'],
#     }
    
with open(f'{out_dir}/results.pkl', "rb") as dill_file:
    results = dill.load(dill_file)
    
AEPs = []
boundary_cons = []
spacing_cons = []
indices = []

for i, result in enumerate(results):
    AEPs.append(result['AEP_final'])
    boundary_cons.append(result['boundary_con'])
    spacing_cons.append(result['spacing_con'])
    indices.append(i)
    
AEPs = np.array(AEPs)
boundary_cons = np.array(boundary_cons)
spacing_cons = np.array(spacing_cons)
indices = np.array(indices)

mask = (boundary_cons < tol) & (spacing_cons < tol)
feasible_AEPs = AEPs[mask]
feasible_indices = indices[mask]

sorted_indices = np.argsort(-feasible_AEPs)
feasible_AEPs = feasible_AEPs[sorted_indices]
feasible_indices = feasible_indices[sorted_indices]

for row in zip(feasible_AEPs[:5], feasible_indices[:5]):
    print(row)
