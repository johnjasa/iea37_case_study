import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import time
import optimization_pyopt as opt
import layout as layout
import matplotlib.pyplot as plt


np.random.seed(314)

file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

def create_initial_optimized_layout(nturbs, file_name_turb, file_name_boundary):
    
    locs = np.zeros((np.sum(nturbs), 2))
    rand = layout.Layout(file_name_turb, file_name_boundary)
    model = layout.Layout(file_name_turb, file_name_boundary)

    counter = 0
    
    for j, nturb in enumerate(nturbs):
        model.re_init()
        rand.re_init()
        
        init_turb_rand_locs = model.add_random_turbine_in_polygon_1(model.polygons[j])

        rand.set_initial_turbine_location(tx=init_turb_rand_locs[0].tolist(), ty=init_turb_rand_locs[1].tolist())
        model.set_initial_turbine_location(tx=init_turb_rand_locs[0].tolist(), ty=init_turb_rand_locs[1].tolist())

        for i in range(nturb):
            turb_rand_locs = rand.add_random_turbine_in_polygon(model.polygons[j])

            turb_locs = model.optimize_individual_turbine(i+1, turb_rand_locs, model.polygons[j])
            
            locs[counter, :] = turb_locs.copy()
            counter += 1
            
    return model, locs

nturbs = [26, 12, 20, 14, 9]

model, locs = create_initial_optimized_layout(nturbs, file_name_turb, file_name_boundary)

model.place_turbines_from_smart_starts(locs[:, 0], locs[:, 1])

AEP = -model._get_AEP_opt()
print('Should be 0:', 2830903.3158824597 - AEP)
model.plot_layout_opt_results()