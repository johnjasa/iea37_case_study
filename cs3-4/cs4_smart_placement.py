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


file_name_turb = 'iea37-ex-opt4_cs4windrose.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'
out_dir = 'tmp'

seed = 314

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model = layout.Layout(file_name_turb, file_name_boundary, wd_step=8)

results = []

class JustPlaceTurbines(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('model')
        self.iteration = 0

    def setup(self):
        self.add_input('turbine_distribution', val=np.array((16, 16, 16, 16)))
        self.add_input('coeff', val=np.ones((5)) * 0.15)
        self.add_input('offset', val=np.ones((5)) * 2.)

        self.add_output('AEP', val=1.0)

    def compute(self, inputs, outputs):
        model = self.options['model']
        turbine_distribution = inputs['turbine_distribution']
        
        # We need this try-except block because the GA might try
        # a distribution of turbines that isn't actually possible.
        # In that case, we return a positive value for AEP so the GA
        # steers away from that point.
        # Over the course of an optimization, the GA should avoid these
        # trouble spots more and more.
        try:
            model.place_turbines_smartly([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])], inputs['coeff'], inputs['offset'])
        except:
            outputs['AEP'] = 1e6
            return
        
        locsx = model.x0
        locsy = model.y0
        
        locsx = model._unnorm(locsx, model.bndx_min, model.bndx_max)
        locsy = model._unnorm(locsy, model.bndy_min, model.bndy_max)
        
        # We don't perform any optimization here; just grab the AEP
        # from this initial layout
        AEP_final = -model._get_AEP_opt()
        
        # We only need to save off the locations and the AEP here since
        # we're not concerned about uthe constraints. The initial layouts
        # should generally satisfy the constraints, but might be a little
        # bit close on the spacing constraint sometimes.
        results_dict = {
            'AEP_final' : AEP_final,
            'locsx' : locsx,
            'locsy' : locsy,
            }
            
        results.append(results_dict)
        
        self.iteration += 1
        fail = False
        
        outputs['AEP'] = -AEP_final
        print()
        print(f'AEP: {AEP_final:.0f}')
        print([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])])
        
        # Only save results every 300 iterations because it takes some time to
        # write to file.
        if self.iteration % 300 == np.floor(self.iteration / 300):
            print('Saving!')
            with open(f'{out_dir}/results.pkl', "wb") as dill_file:
                dill.dump(results, dill_file)

prob = om.Problem()

ivc = prob.model.add_subsystem('ivc', om.IndepVarComp('turbine_distribution', np.array((26, 12, 20, 14))), promotes=['*'])
ivc.add_output('coeff', val=np.ones((5)) * 0.15)
ivc.add_output('offset', val=np.ones((5)) * 2.)
prob.model.add_subsystem('comp', JustPlaceTurbines(model=model), promotes=['*'])

lower = [8, 8, 8, 8]
upper = [40, 40, 40, 40]

prob.model.add_design_var('turbine_distribution', lower=lower, upper=upper)
prob.model.add_design_var('coeff', lower=0.05, upper=0.5)
prob.model.add_design_var('offset', lower=1., upper=10.)
prob.model.add_objective('AEP')

prob.driver = om.SimpleGADriver()
prob.driver.options['pop_size'] = 30
prob.driver.options['max_gen'] = 500
prob.driver.options['bits'] = {'coeff' : 8,
                               'offset' : 8}

prob.setup()
prob.run_driver()