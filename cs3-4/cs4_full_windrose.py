"""
This file performs gradient-based optimization on good initial turbine layouts
using the full cs4 windrose.

After you run `cs4_smart_placement.py`, those results are used at starting
layouts for this script. Basically, the top initial layouts from there are used
as starting points for fine-tuning via gradient-based optimization. Here, we
enforce all constraints and use the full cs4 windrose.
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
start_dir = 'cs4_smart_placement_results'
out_dir = 'cs4_full_windrose_results'

# Only run 50 iterations because we should be relatively close to an optimum
opt_options = {'Major iterations limit': 50,
               'Verify level' : -1}
               
# Number of the top initial layouts to check
num_starts = 20

seed = 314
np.random.seed(seed)

# Create the output directory if it doesn't exist already
try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

# Construct the actual alyout model
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

# Loop through the highest AEP cases, starting with the best
for i in indices[:num_starts]:
    
    # Input the locations from the good initial case into a smart placement algo
    print('Starting AEP:', prev_results[i]['AEP_final'])
    locsx = prev_results[i]['locsx']
    locsy = prev_results[i]['locsy']
    
    model.place_turbines_from_smart_starts(locsx, locsy)

    model.AEP_initial = -model._get_AEP_opt()
    print('Starting AEP after placement (should be the same):', model.AEP_initial)

    # Actually perform optimization
    opt_prob = opt.Optimization(model=model, solver='SNOPT', optOptions=opt_options)

    sol = opt_prob.optimize()

    locsx = sol.getDVs()['x']
    locsy = sol.getDVs()['y']

    locs = np.vstack((locsx, locsy)).T

    locsx = model._unnorm(locsx, model.bndx_min, model.bndx_max)
    locsy = model._unnorm(locsy, model.bndy_min, model.bndy_max)

    AEP_final = -model._get_AEP_opt()

    cons = model.compute_cons({}, locs)

    # Save off results for postprocessing
    results_dict = {
        'optTime' : sol.optTime,
        'optInform' : sol.optInform,
        'AEP_initial' : model.AEP_initial,
        'AEP_final' : AEP_final,
        'locsx' : locsx,
        'locsy' : locsy,
        'boundary_con' : cons['boundary_con'],
        'spacing_con' : cons['spacing_con'],
        }

    results.append(results_dict)

    # Create an image of the layouts and save the optimization history file
    model.plot_layout_opt_results(sol, f'{out_dir}/case_{i}.png')
    shutil.copyfile('SNOPT_print.out', f'{out_dir}/SNOPT_print_{i}.out')

    with open(f'{out_dir}/results.pkl', "wb") as dill_file:
        dill.dump(results, dill_file)
