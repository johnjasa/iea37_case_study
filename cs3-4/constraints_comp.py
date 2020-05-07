import numpy as np
import openmdao.api as om


class ConstraintsComp(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('opt_object')
        
    def setup(self):
        opt_object = self.options['opt_object']
        nturbs = opt_object.nturbs
        
        # Inputs
        self.add_input('locs', val=np.zeros(2*nturbs))
        
        self.add_output('distance')
        self.add_output('space')
        
        self.declare_partials('*', '*', method='fd')
        
    def compute(self, inputs, outputs):
        opt_object = self.options['opt_object']
        
        outputs['distance'][:] = opt_object._distance_from_boundaries(inputs['locs'], opt_object.boundaries_norm)
        outputs['space'][:] = opt_object._space_constraint(inputs['locs'], opt_object.min_dist)
        # These values have to positive
        
        