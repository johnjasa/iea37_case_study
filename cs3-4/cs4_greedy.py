import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import time
import optimization_pyopt as opt
import layout as layout
import matplotlib.pyplot as plt


file_name_turb = 'iea37-ex-opt4.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

opt_options = {'Major iterations limit': 100,
               'Verify level' : -1}
out_dir = 'cs4_GA_results'
# seed = 157

# np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

def create_initial_optimized_layout(nturbs, file_name_turb, file_name_boundary):

    for j in range(len(nturbs)):
        init_turb = layout.Layout(file_name_turb, file_name_boundary)
        rand = layout.Layout(file_name_turb, file_name_boundary)
        model = layout.Layout(file_name_turb, file_name_boundary)

        init_turb_rand_locs = init_turb.add_random_turbine_in_polygon(model.polygons[j])

        rand.set_initial_turbine_location(tx=init_turb_rand_locs[0].tolist(), ty=init_turb_rand_locs[1].tolist())
        model.set_initial_turbine_location(tx=init_turb_rand_locs[0].tolist(), ty=init_turb_rand_locs[1].tolist())

        # nturbs = 10
        time_opt = np.zeros(nturbs[j])

        grid = plt.GridSpec(1, 2, wspace=0.4, hspace=0.3)

        for i in range(nturbs[j]):
            turb_rand_locs = rand.add_random_turbine_in_polygon(model.polygons[j])

            print('2) Greedy network-based optimization...')
            t1 = time.time()

            turb_locs = model.optimize_individual_turbine(i+1, turb_rand_locs, model.polygons[j])
            # model.update_network(turb_locs, i+1)

            t2 = time.time()
            time_opt[i] = t2 - t1

            print('Total time = ', np.sum(time_opt[i]))

        plt.figure()
        plt.subplot(grid[0,0])
        # rand.plot_turbines(label='random', c='ro',fill=False)
        rand.plot_turbines(c='ro', fill=False)
        model.plot_boundaries()
        plt.title('Initial Condition')
        plt.axis('equal')

        # plot network for greedy case
        plt.subplot(grid[0,1])
        # model.plot_network()
        # model.plot_turbines(label='NB-greedy',c='go',fill=False)
        model.plot_turbines(c='go', fill=False)
        plt.title('Network-Based Greedy')
        model.plot_boundaries()
        plt.axis('equal')

# model.plot_turbines(label='NB-greedy',c='go',fill=False)
# model.plot_boundaries()

# axs = plt.gca()
# axs.set_aspect('equal', 'box')

nturbs = [26, 12, 20, 14, 19]

create_initial_optimized_layout(nturbs, file_name_turb, file_name_boundary)

plt.show()
