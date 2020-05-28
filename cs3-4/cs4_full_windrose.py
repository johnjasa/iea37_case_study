import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

opt_options = {'Major iterations limit': 50,
               'Verify level' : -1}
out_dir = 'cs4_full_windrose_results'
start_dir = 'cs4_smart_placement_results'
num_starts = 20

seed = 314

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model = layout.Layout(file_name_turb, file_name_boundary)

results = []

with open(f'{start_dir}/results.pkl', "rb") as dill_file:
    prev_results = dill.load(dill_file)
    
AEPs = []
indices = []

for i, result in enumerate(prev_results):
    AEPs.append(result['AEP_final'])
    indices.append(i)
    
AEPs = np.array(AEPs)
indices = np.array(indices)

sorted_indices = np.argsort(-AEPs)
AEPs = AEPs[sorted_indices]
indices = indices[sorted_indices]

for i in indices[:num_starts]:
    
    print('Starting AEP:', prev_results[i]['AEP_final'])
    locsx = prev_results[i]['locsx']
    locsy = prev_results[i]['locsy']
    
    model.place_turbines_from_smart_starts(locsx, locsy)

    model.AEP_initial = -model._get_AEP_opt()
    print('Starting AEP after placement (should be the same):', model.AEP_initial)

    opt_prob = opt.Optimization(model=model, solver='SNOPT', optOptions=opt_options)

    sol = opt_prob.optimize()

    locsx = sol.getDVs()['x']
    locsy = sol.getDVs()['y']

    locs = np.vstack((locsx, locsy)).T

    locsx = model._unnorm(locsx, model.bndx_min, model.bndx_max)
    locsy = model._unnorm(locsy, model.bndy_min, model.bndy_max)

    AEP_final = -model._get_AEP_opt()

    cons = model.compute_cons({}, locs)

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

    model.plot_layout_opt_results(sol, f'{out_dir}/case_{i}.png')
    shutil.copyfile('SNOPT_print.out', f'{out_dir}/SNOPT_print_{i}.out')

    with open(f'{out_dir}/results.pkl', "wb") as dill_file:
        dill.dump(results, dill_file)
