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

lkj

results = []

class GradientOpt(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('model')
        self.iteration = 0

    def setup(self):
        self.add_input('turbine_distribution', val=np.array((16, 16, 16, 16)))

        self.add_output('AEP', val=1.0)

    def compute(self, inputs, outputs):
        model = self.options['model']
        turbine_distribution = inputs['turbine_distribution']
        
        try:
            # model.place_turbines_within_bounds([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])])
            # model.place_turbines_along_bounds([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])])
            model.place_turbines_smartly([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])])
        except ValueError:
            outputs['AEP'] = 1e6
            return
        
        model.AEP_initial = -model._get_AEP_opt()

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
        
        model.plot_layout_opt_results(sol, f'{out_dir}/case_{self.iteration}.png')
        shutil.copyfile('SNOPT_print.out', f'{out_dir}/SNOPT_print_{self.iteration}.out')
        
        self.iteration += 1
        fail = False
        
        outputs['AEP'] = -AEP_final
        
        with open(f'{out_dir}/results.pkl', "wb") as dill_file:
            dill.dump(results, dill_file)

prob = om.Problem()

prob.model.add_subsystem('ivc', om.IndepVarComp('turbine_distribution', np.array((26, 14, 14, 16))), promotes=['*'])
prob.model.add_subsystem('comp', GradientOpt(model=model), promotes=['*'])

lower = [20, 12, 12, 16]
upper = [28, 20, 18, 24]

prob.model.add_design_var('turbine_distribution', lower=lower, upper=upper)
prob.model.add_objective('AEP')

prob.driver = om.SimpleGADriver()
prob.driver.options['debug_print'] = ['desvars', 'objs']
prob.driver.options['pop_size'] = 8

prob.setup()
prob.run_driver()