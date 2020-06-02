import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

start_dir = 'cs4_smart_placement_results_new'
num_starts = 50

wd_steps = [36, 24, 18, 10, 8, 6, 4, 2]

print('Avg coarse AEP   Avg full AEP   Avg absolute diff   Degrees between bins')

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
    
for i_step, wd_step in enumerate(wd_steps):
    model_coarse = layout.Layout(file_name_turb, file_name_boundary, wd_step=wd_step)
    
    if i_step == 0:
        model_full = layout.Layout(file_name_turb, file_name_boundary)

    if i_step == 0:
        AEP_full = 0
        
    AEP_coarse = 0
    diff = 0
    
    for i in indices[:num_starts]:
        locsx = results[i]['locsx']
        locsy = results[i]['locsy']
        
        model_coarse.place_turbines_from_smart_starts(locsx, locsy)
        AEP_coarse -= model_coarse._get_AEP_opt()
        
        if i_step == 0:
            model_full.place_turbines_from_smart_starts(locsx, locsy)
            AEP_full -= model_full._get_AEP_opt()
        
    AEP_coarse /= num_starts
    
    if i_step == 0:
        AEP_full /= num_starts
    
    diff = np.abs(AEP_coarse - AEP_full)

    print(f'       {AEP_coarse:.0f}        {AEP_full:.0f}              {diff:6.0f}                   {wd_step}')