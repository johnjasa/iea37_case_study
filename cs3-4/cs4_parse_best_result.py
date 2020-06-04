"""
This file plots the best result and does any postprocessing as needed.
"""

import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


# Set the turbine and boundary files; make sure we're using the full windrose
file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

# List the directory with results from `cs4_smart_placement.py` and the
# directory where you want outputs for this file
start_dir = 'cs4_full_windrose_results_best'
out_dir = 'cs4_full_windrose_results_best'

# Only run 50 iterations because we should be relatively close to an optimum
opt_options = {'Major iterations limit': 0,
               'Verify level' : -1,
               'Major optimality tolerance' : 5e-3}
               
# Construct the actual layout model
model = layout.Layout(file_name_turb, file_name_boundary)

# Read in results from the initial turbine placement algo
with open(f'{start_dir}/results.pkl', "rb") as dill_file:
    prev_results = dill.load(dill_file)
    
AEPs = []
indices = []

# Create lists of the AEPs and the corresponding indices
for i, result in enumerate(prev_results):
    AEPs.append(result['AEP_final'])
    indices.append(i)

# Obtain the indices of the highest AEP cases    
AEPs = np.array(AEPs)
indices = np.array(indices)
sorted_indices = np.argsort(-AEPs)
AEPs = AEPs[sorted_indices]
indices = indices[sorted_indices]

results = []

num_starts = 1

# Loop through the highest AEP cases, starting with the best
for i in indices[:num_starts]:
    print()
    
    # Input the locations from the good initial case into a smart placement algo
    print('Starting AEP:', prev_results[i]['AEP_final'])
    locsx = prev_results[i]['locsx']
    locsy = prev_results[i]['locsy']
    
    model.place_turbines_from_smart_starts(locsx, locsy)
    
    locsx_ = model._norm(locsx, model.bndx_min, model.bndx_max)
    locsy_ = model._norm(locsy, model.bndy_min, model.bndy_max)
    
    locs = np.vstack((locsx_, locsy_)).T
    
    model.AEP_initial = -model._get_AEP_opt()
    print('Starting AEP after placement:', model.AEP_initial)

    AEP_final = -model._get_AEP_opt()

    cons = model.compute_cons({}, locs)

    # Save off results for postprocessing
    results_dict = {
        'AEP_initial' : model.AEP_initial,
        'AEP_final' : AEP_final,
        'locsx' : locsx,
        'locsy' : locsy,
        'boundary_con' : cons['boundary_con'],
        'spacing_con' : cons['spacing_con'],
        }
        
    print(np.max(cons['boundary_con']))

    results.append(results_dict)

    # Create an image of the layouts and save the optimization history file
    model.plot_layout_opt_results(filename=f'{out_dir}/final_layout.pdf', final_result=True)
    
    locs = np.vstack((locsx, locsy)).T
    for row in locs:
        print(f'      - [{row[0]:.8f},  {row[1]:.8f}]')
