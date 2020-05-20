import os
import shutil
import optimization_pyopt as opt
import layout as layout
import numpy as np
import dill


file_name_turb = 'two_boxes_layout.yaml'
file_name_boundary = 'two_boxes_boundaries.yaml'

opt_options = {'Major iterations limit': 50}
out_dir = 'two_boxes_results'
seed = 314
repetitions = 5

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model = layout.Layout(file_name_turb, file_name_boundary)

results = []

for i_rep in range(repetitions):
    for i_case in range(3, 7):
        model.place_turbines_within_bounds([i_case, model._nturbs - i_case])
        model.AEP_initial = -model._get_AEP_opt()

        opt_prob = opt.Optimization(model=model, solver='SNOPT', optOptions=opt_options)
        
        sol = opt_prob.optimize()

        print(sol)
        
        locsx = sol.getDVs()['x']
        locsy = sol.getDVs()['y']

        locsx = model._unnorm(locsx, model.bndx_min, model.bndx_max)
        locsy = model._unnorm(locsy, model.bndy_min, model.bndy_max)
        
        results_dict = {
            'optTime' : sol.optTime,
            'optInform' : sol.optInform,
            'AEP_initial' : model.AEP_initial,
            'AEP_final' : -model._get_AEP_opt(),
            'locsx' : locsx,
            'locsy' : locsy,
            }
        
        results.append(results_dict)
        print(results_dict)

        model.plot_layout_opt_results(sol, f'{out_dir}/case_{i_case}_{i_rep}.png')
    
with open(f'{out_dir}/results.pkl', "wb") as dill_file:
    dill.dump(results, dill_file)