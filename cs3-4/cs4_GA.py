import os
import shutil
import openmdao.api as om
import numpy as np
import dill
import optimization_pyopt as opt
import layout as layout


file_name_turb = 'iea37-ex-opt4.yaml'
file_name_boundary = 'iea37-boundary-cs4.yaml'

opt_options = {'Major iterations limit': 100,
               'Verify level' : -1}
out_dir = 'cs4_GA_results'
seed = 314

np.random.seed(seed)

try:
    os.mkdir(out_dir) 
except FileExistsError:
    shutil.rmtree(out_dir)
    os.mkdir(out_dir) 

model = layout.Layout(file_name_turb, file_name_boundary)

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
        
        model.place_turbines_within_bounds([turbine_distribution[0], turbine_distribution[1], turbine_distribution[2], turbine_distribution[3], model._nturbs - sum(turbine_distribution[:4])])
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
        
        self.iteration += 1
        fail = False
        
        outputs['AEP'] = -AEP_final
        
        with open(f'{out_dir}/results.pkl', "wb") as dill_file:
            dill.dump(results, dill_file)

prob = om.Problem()

prob.model.add_subsystem('ivc', om.IndepVarComp('turbine_distribution', np.array((16, 16, 16, 16))), promotes=['*'])
prob.model.add_subsystem('comp', GradientOpt(model=model), promotes=['*'])

prob.model.add_design_var('turbine_distribution', lower=10, upper=22)
prob.model.add_objective('AEP')

prob.driver = om.SimpleGADriver()
prob.driver.options['debug_print'] = ['desvars', 'objs']
prob.driver.options['pop_size'] = 8

prob.setup()
prob.run_driver()
    
with open(f'{out_dir}/results.pkl', "wb") as dill_file:
    dill.dump(results, dill_file)