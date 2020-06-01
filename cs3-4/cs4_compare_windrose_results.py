import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

out_dir = 'windrose_comparisons'
start_dir = 'cs4_smart_placement_results_new'
num_starts = 10

seed = 314

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model_coarse = layout.Layout(file_name_turb, file_name_boundary, wd_step=2)
model_full = layout.Layout(file_name_turb, file_name_boundary)

results = []

with open(f'{start_dir}/results.pkl', "rb") as dill_file:
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

print('Coarse AEP     Full AEP      Coarse-Full diff')
for i in indices[:num_starts]:
    locsx = results[i]['locsx']
    locsy = results[i]['locsy']
    
    model_coarse.place_turbines_from_smart_starts(locsx, locsy)
    model_coarse.AEP_initial = -model_coarse._get_AEP_opt()
    
    model_full.place_turbines_from_smart_starts(locsx, locsy)
    model_full.AEP_initial = -model_full._get_AEP_opt()

    print(f'{model_coarse.AEP_initial:.0f}        {model_full.AEP_initial:.0f}       {model_coarse.AEP_initial-model_full.AEP_initial:.0f}')