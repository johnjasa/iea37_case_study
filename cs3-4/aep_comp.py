import numpy as np
import openmdao.api as om


class AEPComp(om.ExplicitComponent):
    # Openmdao component to parameterize distributed quantities for the outer shape of the wind turbine rotor blades
    def initialize(self):
        self.options.declare('opt_object')
        
    def setup(self):
        opt_object = self.options['opt_object']
        nturbs = opt_object.nturbs
        
        # Inputs
        self.add_input('locs', val=np.zeros(2*nturbs))
        
        self.add_output('AEP')
        
        self.declare_partials('*', '*')
        
    def compute(self, inputs, outputs):
        outputs['AEP'] = self.options['opt_object']._get_AEP_opt(inputs['locs'])
        
    def compute_partials(self, inputs, partials):
        partials['AEP', 'locs'] = self.options['opt_object'].grad(inputs['locs'])
        
    