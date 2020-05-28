import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


out_dir = 'cs4_smart_placement_results_new'
num_best = 30
    
with open(f'{out_dir}/results.pkl', "rb") as dill_file:
    results = dill.load(dill_file)
    
AEPs = []
indices = []

for i, result in enumerate(results):
    AEPs.append(result['AEP_final'])
    indices.append(i)
    
AEPs = np.array(AEPs)
indices = np.array(indices)

sorted_indices = np.argsort(-AEPs)
AEPs = AEPs[sorted_indices]
indices = indices[sorted_indices]

for row in zip(AEPs[:num_best], indices[:num_best]):
    print(row)
