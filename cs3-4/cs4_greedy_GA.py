"""
This file generates the best initial layouts for the cs4 problem.

A GA controls the number of turbines in each region and then a relatively
naive placement algorithm places turbines in each of the regions, mostly along
the borders and some in the interior areas of the regions. This GA runs for many
iterations and should roughly converge to only reasonably good layouts.

The results from this script are then used in `cs4_local_optimization.py`, which
takes the best layouts from here and uses those as starting points for
gradient-based optimization to really fine-tune the results.

"""

import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout
from time import time


file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'
out_dir = 'cs4_greedy_results'

seed = 314

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model = layout.Layout(file_name_turb, file_name_boundary)  #, wd_step=8)
rand = layout.Layout(file_name_turb, file_name_boundary)  #, wd_step=8)

results = []

class JustPlaceTurbines(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('model')
        self.options.declare('rand')
        self.iteration = 0

    def setup(self):
        self.add_input('turbine_distribution', val=np.array((16, 16, 16, 16)))
        self.add_input('coeff', val=np.ones((5)) * 0.15)
        self.add_input('offset', val=np.ones((5)) * 2.)

        self.add_output('AEP', val=1.0)

    def compute(self, inputs, outputs):
        model = self.options['model']
        rand = self.options['rand']
        turbine_distribution = inputs['turbine_distribution'].copy()
        turbine_distribution = [turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])]
        turbine_distribution = [int(i) for i in turbine_distribution]
        print(turbine_distribution)
        
        # If the GA gives an invalid number of turbines, return with a penalized AEP value
        if turbine_distribution[-1] < 0:
            outputs['AEP'] = 1e6
            return

        counter = 0
        
        s = time()
        
        try:
            locs = np.zeros((model._nturbs, 2))
            for j, nturb in enumerate(turbine_distribution):
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
                    
        except ValueError:
            print('Could not find space for a turbine')
            outputs['AEP'] = 1e6
            return
                
        model.place_turbines_from_smart_starts(locs[:, 0], locs[:, 1])
        
        # We don't perform any optimization here; just grab the AEP
        # from this initial layout
        AEP_final = -model._get_AEP_opt()
        
        print(time() - s, ' secs')
        
        # We only need to save off the locations and the AEP here since
        # we're not concerned about uthe constraints. The initial layouts
        # should generally satisfy the constraints, but might be a little
        # bit close on the spacing constraint sometimes.
        results_dict = {
            'AEP_final' : AEP_final,
            'locsx' : locs[:, 0],
            'locsy' : locs[:, 1],
            }
            
        results.append(results_dict)
        
        self.iteration += 1
        
        outputs['AEP'] = -AEP_final
        print(f'AEP: {AEP_final:.0f}')
        
        # Only save results every 50 iterations because it takes some time to
        # write to file.
        if self.iteration % 50 == np.floor(self.iteration / 50):
            model.plot_layout_opt_results(filename=f'{out_dir}/case_{self.iteration}.png')
            print('Saving!')
            with open(f'{out_dir}/results.pkl', "wb") as dill_file:
                dill.dump(results, dill_file)

prob = om.Problem()

ivc = prob.model.add_subsystem('ivc', om.IndepVarComp('turbine_distribution', np.array((26, 12, 20, 14))), promotes=['*'])
prob.model.add_subsystem('comp', JustPlaceTurbines(model=model, rand=rand), promotes=['*'])

lower = [12, 12, 12, 12]
upper = [32, 32, 32, 32]

prob.model.add_design_var('turbine_distribution', lower=lower, upper=upper)
prob.model.add_objective('AEP')

prob.driver = om.SimpleGADriver()
prob.driver.options['pop_size'] = 20
prob.driver.options['max_gen'] = 500

prob.setup()
prob.run_driver()