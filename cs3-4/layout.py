# Copyright 2020 NREL
 
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0
 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
 
# See https://floris.readthedocs.io for documentation
 
from autograd import grad
import autograd.numpy as np
import matplotlib.pyplot as plt
# import iea37_aepcalc_test as ieatools
import iea37_aepcalc_fast as ieatools
from time import time
from scipy.spatial.distance import cdist
from shapely.geometry import MultiPolygon, MultiPoint, Polygon


class Layout():
    def __init__(self, file_name, bnds_file_name, min_dist=None, 
                                                  jac=False, 
                                                  opt_method='SLSQP', opt_options=None):

        self.file_name = file_name

        self.coords, self.fname_turb, self.fname_wr = \
            ieatools.getTurbLocYAML(self.file_name)
        # print('self.coords: ', self.coords)
        self.cut_in, self.cut_out, self.rated_ws, self.rated_pow, self.D = \
            ieatools.getTurbAtrbtYAML(self.fname_turb)
        self.wd, self.wd_freq, self.ws, self.ws_freq, self.ws_bin_num, self.ws_min, self.ws_max = \
            ieatools.getWindRoseYAML(self.fname_wr)

        self.min_dist = 2*self.rotor_diameter

        self.boundaries = ieatools.getBoundaryAtrbtYAML(bnds_file_name)
        self.bndx_min, self.bndy_min = np.min(np.array(self.boundaries), axis=(0,1))
        self.bndx_max, self.bndy_max = np.max(np.array(self.boundaries), axis=(0,1))
        
        self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
                                self.bndx_max), self._norm(val[1], \
                                self.bndy_min, self.bndy_max)] \
                                for val in self.boundaries]
        self.polygons = MultiPolygon([Polygon(bounds) for bounds in self.boundaries])
        
        self.AEP_initial = self.get_AEP()

        self.set_initial_locs(self._coords_to_locs())
        self.x = self.x0
        self.y = self.y0
        # print('self.coords: ', self.coords)
        
        self.gradient = grad(self._get_AEP_opt)
        self.boundary_con_grad = grad(self.distance_from_boundaries)
        self.space_con_grad = grad(self.space_constraint)
        
    def __str__(self):
        return 'layout'

    def _norm(self, val, x1, x2):
        return (val - x1)/(x2 - x1)
    
    def _unnorm(self, val, x1, x2):
        return np.array(val)*(x2 - x1) + x1

    def set_initial_locs(self, locs):
        self.x0 = [self._norm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0:self.nturbs]]
        self.y0 = [self._norm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[self.nturbs:2*self.nturbs]]
                
    ###########################################################################
    # Required private optimization methods
    ###########################################################################

    def reinitialize(self):
        pass

    # def _sens(self, varDict, funcs):
    #     self.parse_opt_vars(varDict)
    #     locs = [self.x] + [self.y]
    # 
    #     funcsSens = {}
    #     funcsSens['obj', 'x'] = self.gradient(locs)[0]
    #     funcsSens['obj', 'y'] = self.gradient(locs)[1]
    # 
    #     funcsSens['boundary_con', 'x'] = self.boundary_con_grad(locs)[0]
    #     funcsSens['boundary_con', 'y'] = self.boundary_con_grad(locs)[1]
    #     funcsSens['spacing_con', 'x'] = self.space_con_grad(locs)[0]
    #     funcsSens['spacing_con', 'y'] = self.space_con_grad(locs)[1]
    # 
    #     fail = False
    #     return funcsSens, fail



    def obj_func(self, varDict):
        # Parse the variable dictionary
        self.parse_opt_vars(varDict)

        # Update turbine map with turbince locations
        locs = [self.x] + [self.y]

        # Compute the objective function
        funcs = {}
        funcs['obj'] = self._get_AEP_opt(locs)

        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs, locs)

        fail = False
        return funcs, fail

    def _get_AEP_opt(self, locs):
        # print('locs: ', locs)
        locs_tmp = [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0:self.nturbs]] \
                + [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[self.nturbs:2*self.nturbs]]
        self._locs_to_coords(np.array(locs_tmp))
        # print('locs_tmp :', locs_tmp)

        # print('===== before get_AEP =====')
        AEP = self.get_AEP()
        # print('AEP: ', AEP)
        # print('AEP_initial: ', self.AEP_initial)
        # print('===== after get_AEP =====')
        
        return -AEP/self.AEP_initial

    def get_AEP(self):
        AEP = ieatools.calcAEPcs3(self.coords, self.wd_freq, self.ws,
            self.ws_freq, self.wd, self.D, self.cut_in, self.cut_out, 
            self.rated_ws, self.rated_pow)

        # return np.sum(AEP)
        return AEP

    def _locs_to_coords(self, locs):
        # print('here')
        # print('nturbs: ', self.nturbs)
        # print('locs again: ', locs)
        # print('2: ', locs[0])
        self.coords = np.array(list(zip(locs[0], locs[1])))
        # print('1: ', self.coords)

    def _coords_to_locs(self):
        locs = np.concatenate((self.coords[:,0], self.coords[:,1]))
        return locs

    # Optionally, the user can supply the optimization with gradients
    # def _sens(self, varDict, funcs):
    #     funcsSens = {}
    #     fail = False
    #     return funcsSens, fail

    def parse_opt_vars(self, varDict):
        self.x = varDict['x']
        self.y = varDict['y']

    def parse_sol_vars(self, sol):
        self.x = list(sol.getDVs().values())[0]
        self.y = list(sol.getDVs().values())[1]

    def add_var_group(self, optProb):
        # optProb.addVarGroup('x', self.nturbs, type='c',
        #                     lower=self.bndx_min,
        #                     upper=self.bndx_max,
        #                     value=self.x0)
        # optProb.addVarGroup('y', self.nturbs, type='c',
        #                     lower=self.bndy_min,
        #                     upper=self.bndy_max,
        #                     value=self.y0)
        optProb.addVarGroup('x', self.nturbs, type='c',
                            lower=0,
                            upper=1,
                            value=self.x0)
        optProb.addVarGroup('y', self.nturbs, type='c',
                            lower=0,
                            upper=1,
                            value=self.y0)

        return optProb

    def add_con_group(self, optProb):
        optProb.addConGroup('boundary_con', 1, upper=0.0)
        optProb.addConGroup('spacing_con', 1, upper=0.0)

        return optProb

    def compute_cons(self, funcs, locs):
        funcs['boundary_con'] = self.distance_from_boundaries(locs)
        funcs['spacing_con'] = self.space_constraint(locs)

        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################

    def space_constraint(self, locs, rho=50):
        x = [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0]]
        y = [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[1]]
                
        # Sped up distance calc here using vectorization
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)
                
        g = 1 - np.array(dist) / self.min_dist
        
        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)
        
        return KS_constraint[0][0]
        

    def distance_from_boundaries(self, locs, rho=500):  
        x = [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0]]
        y = [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[1]]

        locs = np.vstack((x, y)).T
        points = MultiPoint(locs)
        dist_out = np.zeros((len(points), len(self.polygons)))
        for j, polygon in enumerate(self.polygons):
            for i, point in enumerate(points):
                dist_out[i, j] = polygon.exterior.distance(point)
                if not polygon.contains(point):
                    dist_out[i, j] *= -1.
                    
        # We only care if the point is in one of the regions
        dist_out = -np.max(dist_out, axis=1)
        
        g = dist_out / 1e4
        
        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.ravel(g))
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents)
        KS_constraint = g_max + 1.0 / rho * np.log(summation)
        
        return KS_constraint

    def plot_layout_opt_results(self, sol):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx = sol.getDVs()['x']
        locsy = sol.getDVs()['y']

        locsx = [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in locsx]
        locsy = [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in locsy]

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(
            [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in self.x0],
            [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in self.y0],
            'ob'
        )
        plt.plot(locsx, locsy, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        for polygon in self.polygons:    
            xs, ys = polygon.exterior.xy    
            plt.plot(xs, ys, alpha=0.5, color='b')

        plt.show()

    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS 
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = np.shape(self.coords)[0]
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.D