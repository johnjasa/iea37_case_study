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
 
# from autograd import grad
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.optimize import minimize
# import iea37_aepcalc_test as ieatools
import iea37_aepcalc_fast as ieatools
from time import time
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from shapely.geometry import MultiPolygon, MultiPoint, Polygon, Point


class Layout():
    def __init__(self, file_name, bnds_file_name, min_dist=None, 
                                                  jac=False, 
                                                  opt_method='SLSQP', opt_options=None,
                                                  wd_step=None):

        self.file_name = file_name
        self.bnds_file_name = bnds_file_name

        self.cluster_distance = 5000

        self.tx = []
        self.ty = []

        self.nturbs_initial = 0

        self.coords, self.fname_turb, self.fname_wr = \
            ieatools.getTurbLocYAML(self.file_name)
        self.cut_in, self.cut_out, self.rated_ws, self.rated_pow, self.D = \
            ieatools.getTurbAtrbtYAML(self.fname_turb)
        self.wd, self.wd_freq, self.ws, self.ws_freq, self.ws_bin_num, self.ws_min, self.ws_max = \
            ieatools.getWindRoseYAML(self.fname_wr)
            
        if wd_step is not None:
            self.wd, self.wd_freq, self.ws_freq = self.resample_windrose(wd_step)

        self.A = np.matrix([0])  # adjacency matrix - initially only one node

        if min_dist is None:
            self.min_dist = 2*self.rotor_diameter
        else:
            self.min_dist = min_dist

        self.boundaries = ieatools.getBoundaryAtrbtYAML(bnds_file_name)
        bound_array = np.vstack(self.boundaries)
        self.bndx_min, self.bndy_min = np.min(bound_array, axis=0)
        self.bndx_max, self.bndy_max = np.max(bound_array, axis=0)
        
        self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
                                self.bndx_max), self._norm(val[1], \
                                self.bndy_min, self.bndy_max)] \
                                for val in self.boundaries]
        self.polygons = MultiPolygon([Polygon(bounds) for bounds in self.boundaries])
        
        self.AEP_initial = self.get_AEP()

        self.set_initial_locs(self._coords_to_locs())
        self.x = self.x0
        self.y = self.y0
        
    def re_init(self):
        self.tx = []
        self.ty = []
        self.nturbs_initial = 0
        self.A = np.matrix([0])  # adjacency matrix - initially only one node
        
    def __str__(self):
        return 'layout'

    def _norm(self, val, x1, x2):
        return (np.array(val) - x1)/(x2 - x1)
    
    def _unnorm(self, val, x1, x2):
        return np.array(val)*(x2 - x1) + x1

    def set_initial_locs(self, locs):
        self.x0 = self._norm(locs[0:self.nturbs], self.bndx_min, self.bndx_max)
        self.y0 = self._norm(locs[self.nturbs:2*self.nturbs], self.bndy_min, self.bndy_max)
        
    def resample_windrose(self, wd_step):
        num_bins = 360 // wd_step
        new_wind_dir = np.arange(0, 360, wd_step)
        new_wind_dir_freq = np.zeros(num_bins)
        
        for i in range(num_bins):
            bin_center = int(new_wind_dir.take([i], mode='wrap'))
            bin_0 = bin_center - wd_step // 2
            bin_1 = bin_center + wd_step // 2
            
            indices = range(bin_0, bin_1)
            new_wind_dir_freq[i] = np.sum(self.wd_freq.take(indices, mode='wrap'))
            
        new_wind_speed_probs = np.zeros((num_bins, len(self.ws)))
        
        for i in range(num_bins):
            bin_center = int(new_wind_dir.take([i], mode='wrap'))
            bin_0 = bin_center - wd_step // 2
            bin_1 = bin_center + wd_step // 2
            
            indices = range(bin_0, bin_1)
            wind_probs = self.ws_freq.take(indices, axis=0, mode='wrap')
            summed_wind_probs = np.sum(wind_probs, axis=0)
            new_wind_speed_probs[i, :] = summed_wind_probs / np.linalg.norm(summed_wind_probs, ord=1)
            
        return new_wind_dir, new_wind_dir_freq, new_wind_speed_probs
                
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

        # Compute the objective function
        funcs = {}
        funcs['obj'] = self._get_AEP_opt()
        
        locs = np.vstack((self.x, self.y)).T
        
        # Compute constraints, if any are defined for the optimization
        funcs = self.compute_cons(funcs, locs)

        fail = False
        return funcs, fail

    def _get_AEP_opt(self):
        x = self._unnorm(self.x, self.bndx_min, self.bndx_max)
        y = self._unnorm(self.y, self.bndy_min, self.bndy_max)
        self.coords = np.vstack((x, y)).T

        # print('===== before get_AEP =====')
        AEP = self.get_AEP()
        # print('AEP: ', AEP)
        # print('AEP_initial: ', self.AEP_initial)
        # print('===== after get_AEP =====')
        
        return -AEP
        
    def get_AEP(self):
        AEP = ieatools.calcAEPcs3(self.coords, self.wd_freq, self.ws,
            self.ws_freq, self.wd, self.D, self.cut_in, self.cut_out, 
            self.rated_ws, self.rated_pow)

        # return np.sum(AEP)
        return AEP

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
        optProb.addConGroup('boundary_con', self.nturbs, upper=0.0)
        optProb.addConGroup('spacing_con', 1, upper=0.0)

        return optProb

    def compute_cons(self, funcs, locs):
        funcs['boundary_con'] = self.distance_from_boundaries(locs)
        funcs['spacing_con'] = self.space_constraint(locs)

        return funcs

    ###########################################################################
    # User-defined methods
    ###########################################################################
    
    def randomly_place_turbines(self, seed=314):
        """ This will place turbines entirely randomly with no notion of
        feasbility or being within bounds. """
        np.random.seed(seed)
        x = np.random.random(self._nturbs)
        y = np.random.random(self._nturbs)
        self.x, self.y = x, y
        
    def generate_random(self, number, polygon):
        list_of_points = []
        minx, miny, maxx, maxy = polygon.bounds
        counter = 0
        while counter < number:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if polygon.contains(pnt):
                list_of_points.append(pnt.coords)
                counter += 1
        return list_of_points

    def set_initial_turbine_location(self, tx, ty):
        self.tx = tx
        self.ty = ty
        self.nturbs_initial += 1

    def add_random_turbine_in_polygon(self, polygon):
        # add turbine to simulation
        count = 0
        dist = 0
        turb_in_poly = False

        poly_xmin = polygon.bounds[0]
        poly_ymin = polygon.bounds[1]
        poly_xmax = polygon.bounds[2]
        poly_ymax = polygon.bounds[3]

        while (dist < self.min_dist) or (turb_in_poly == False):
            x = np.random.rand(1) * (poly_xmax - poly_xmin) + poly_xmin
            y = np.random.rand(1) * (poly_ymax - poly_ymin) + poly_ymin
            pnt = Point(x, y)

            # if first turbine, set dist value so minimum spacing is true
            if len(self.tx) < 1:
                dist = 2 * self.min_dist
            else:
                dist = np.min(np.sqrt( (x-self.tx)**2 + (y-self.ty)**2 ))

            turb_in_poly = polygon.contains(pnt)

            count = count + 1

            if count == 1000:
                raise ValueError('Cannot find space for a turbine')

        # add node to the system
        self.tx.append(x[0])
        self.ty.append(y[0])
        self.nturbs_initial = self.nturbs_initial + 1

        return x, y
        
    def add_random_turbine_in_polygon_1(self, polygon):
        # add turbine to simulation
        count = 0
        dist = 0
        turb_in_poly = False

        poly_xmin = polygon.bounds[0]
        poly_ymin = polygon.bounds[1]
        poly_xmax = polygon.bounds[2]
        poly_ymax = polygon.bounds[3]

        while (dist < self.min_dist) or (turb_in_poly == False):
            x = np.random.rand(1) * (poly_xmax - poly_xmin) + poly_xmin
            y = np.random.rand(1) * (poly_ymax - poly_ymin) + poly_ymin
            pnt = Point(x, y)

            # if first turbine, set dist value so minimum spacing is true
            if len(self.tx) < 1:
                dist = 2 * self.min_dist
            else:
                dist = np.min(np.sqrt( (x-self.tx)**2 + (y-self.ty)**2 ))

            turb_in_poly = polygon.contains(pnt)

            count = count + 1

            if count == 1000:
                raise ValueError('Cannot find space for a turbine')

        return x, y

    def optimize_individual_turbine(self, turb_id, turb_loc, polygon):
        # TODO: dynamic programming and or mixed integer linear program

        # update number of turbines
        self.nturbs_initial = self.nturbs_initial + 1

        self.poly_xmin = polygon.bounds[0]
        self.poly_ymin = polygon.bounds[1]
        self.poly_xmax = polygon.bounds[2]
        self.poly_ymax = polygon.bounds[3]

        # initialize turbine location
        x0 = copy.deepcopy(turb_loc)

        # add bounds
        bnds = [
            (self.poly_xmin, self.poly_xmax),
            (self.poly_ymin, self.poly_ymax)
        ]

        # add constraints
        con1 = {
            'type': 'ineq', 'fun': lambda x: \
                np.min(np.sqrt( (x[0] - self.tx)**2 + (x[1] - self.ty)**2)) \
                    - self.min_dist
        }
        con2 = {
            'type': 'ineq', 'fun' : lambda x, *args: \
                self.distance_from_boundaries_initial(x, polygon), \
                'args': [polygon]
        }
        cons = [con1, con2]

        # optimize location of the one node
        res = minimize(
            self.obj_func_initial_turbine_layout,
            x0,
            args=(turb_id),
            method='SLSQP',
            bounds=bnds,
            constraints=cons
        )

        # add node to the system
        self.tx.append(res.x[0])
        self.ty.append(res.x[1])

        return res.x
    
    def obj_func_initial_turbine_layout(self, x, turb_id):
        # define the new adjacency matrix with added node
        self.update_network(x, turb_id)

        return (1 / np.max(self.wd_freq)) * np.nansum(self.A)

    def update_network(self, x, turb_id):
        tx = np.array(self.tx)
        ty = np.array(self.ty)
        
        # find distance from new node to other nodes
        dist = np.sqrt( (x[0] - tx)**2 + (x[1] - ty)**2 )

        # find turbines that are inside the cluster distance
        indices = np.where(dist <= self.cluster_distance)[0]

        # set up adjacency matrix with additional rows and 
        # columns for the interactions
        tmp_A = np.zeros((self.nturbs_initial, self.nturbs_initial))
        tmp_A[:turb_id, :turb_id] = self.A[:turb_id, :turb_id].copy()
                
        # if there are any nearby turbines update the adjacency matrix
        if len(indices) > 0:
            # identify interactions on the added node
            # find the wind direction that will impact the added turbine
            y_dist = x[1] - ty[indices]
            x_dist = x[0] - tx[indices]
            
            wd = 90 - np.degrees(np.arctan((y_dist / x_dist)))
            wd[wd < 0] += 360
            f = self.freq(wd)
            tmp_A[turb_id, indices] = 10000 * f / dist[indices]
            
            # identify interactions of the added node on other turbines
            y_dist = x[1] - ty[indices]
            x_dist = tx[indices] - x[0]
            
            # get wind direction that is associated with 
            # turbine j impacting turbine i
            wd = 90 - np.degrees(np.arctan((y_dist / x_dist)))
            wd[wd < 0] += 360
            f = self.freq(wd)
            tmp_A[indices, turb_id] = 10000 * f / dist[indices]
                
        np.nan_to_num(tmp_A, nan=10**9)
            
        self.A = copy.deepcopy(tmp_A)

    def freq(self, wd):
        # dist = np.abs(wd - self.wd)
        # idx = np.where(dist == np.min(dist))
        # f = np.sum(self.ws_freq[idx[0]])
        
        # WARNING: this simplification is only possible because
        # the full windrose is 360 degrees
        try:
            f = np.sum(self.ws_freq[wd.astype(int)], axis=1)
        except IndexError:
            print('Handling a nan')
            bad_indices = np.argwhere(np.isnan(wd))
            wd_int = wd.astype(int)
            wd_int[bad_indices] = 0
            f = np.sum(self.ws_freq[wd_int], axis=1)
            f[bad_indices] = np.nan            

        return f

    def distance_from_boundaries_initial(self, locs, polygon, rho=500):
        x = [locs[0]]
        y = [locs[1]]
        
        locs = np.vstack((x, y)).T
        points = MultiPoint(np.array(locs))
        dist_out = np.zeros(len(x))
        for i, point in enumerate(points):
            dist_out[i] = polygon.exterior.distance(point)
            if not polygon.contains(point):
                dist_out[i] *= -1.
                    
        g = dist_out / 1e4
        
        return g
        
    def place_gridded_points(self, number, polygon):
        offset_dist = 2 * self.D
        new_border = polygon.boundary.parallel_offset(offset_dist, 'right')
        new_polygon = Polygon(new_border)
        
        if new_polygon.area > polygon.area:
            new_border = polygon.boundary.parallel_offset(offset_dist, 'left')
            new_polygon = Polygon(new_border)
            
        minx, miny, maxx, maxy = new_polygon.bounds
        
        for num_grid_points in range(10):
            x = np.linspace(minx, maxx, num_grid_points)
            y = np.linspace(miny, maxy, num_grid_points)
            xx, yy = np.meshgrid(x, y)
            x = xx.flatten()
            y = yy.flatten()
            
            list_of_points = []
            for i in range(len(x)):
                pnt = Point(x[i], y[i])
                if new_polygon.contains(pnt):
                    list_of_points.append(pnt.coords)
                    if len(list_of_points) == number:
                        return list_of_points
                        
    def place_turbine_on_inner_bound(self, number, polygon, offset):
        offset_dist = offset * self.D
        new_border = polygon.boundary.parallel_offset(offset_dist, 'right')
        new_polygon = Polygon(new_border)
        
        if new_polygon.area > polygon.area:
            new_border = polygon.boundary.parallel_offset(offset_dist, 'left')
            new_polygon = Polygon(new_border)
            
        boundary = np.array(new_polygon.boundary.coords)
            
        return self.place_turbines_on_bound(boundary, number)
        
    def place_turbines_within_bounds(self, n_sub_turbs, seed=314):
        all_points = []
        for i_polygon, polygon in enumerate(self.polygons):
            points = self.generate_random(n_sub_turbs[i_polygon], polygon)
            all_points.extend(list(points))
            
        all_points = np.array(all_points).squeeze()
        self.x = self._norm(all_points[:, 0], self.bndx_min, self.bndx_max)
        self.y = self._norm(all_points[:, 1], self.bndy_min, self.bndy_max)
        
        self.x0, self.y0 = self.x.copy(), self.y.copy()
        
    def place_turbines_on_bound(self, boundary, n_turbs):
        looped_boundary = np.vstack((boundary, boundary[0]))
        
        # Compute the curvilinear length of the boundary for that region.
        distance = np.cumsum(np.sqrt( np.ediff1d(looped_boundary[:, 0], to_begin=0)**2 + np.ediff1d(looped_boundary[:, 1], to_begin=0)**2 ))
        
        # Compute the parametric curve, normalized from 0 to 1.
        distance = distance / distance[-1]

        # Set up interpolation functions along that parametric curve
        fx, fy = interp1d(distance, looped_boundary[:, 0]), interp1d(distance, looped_boundary[:, 1])
    
        # Create equidistant points in the parametric space with a random
        # starting point
        point_spacings = np.linspace(0, 1, n_turbs, endpoint=False)
        point_spacings += np.random.random()
        point_spacings[point_spacings > 1] -= 1
        
        return fx(point_spacings), fy(point_spacings)
        
    def place_turbines_along_bounds(self, n_sub_turbs):
        
        # Set up vectors for coordinates
        x = self.x.copy()
        y = self.y.copy()
        
        # Loop through each region
        size = 0
        for i_boundary, boundary in enumerate(self.boundaries):
            
            # Get the number of turbines in that region
            n_turbs = int(n_sub_turbs[i_boundary])
            
            new_x, new_y = self.place_turbines_on_bound(boundary, n_turbs)
            
            # Compute the x and y coords for the turbines based off the
            # parametric equidistant points
            x[size:size+n_turbs] = new_x
            y[size:size+n_turbs] = new_y
            
            # Keep track of the total number of turbines that have been placed
            size += n_turbs
            
        # Normalized the coordinates from physical to scaled coordinates
        self.x = self._norm(x, self.bndx_min, self.bndx_max)
        self.y = self._norm(y, self.bndy_min, self.bndy_max)
        
        self.x0, self.y0 = self.x.copy(), self.y.copy()
        
    def place_turbines_smartly(self, n_sub_turbs, coeff, offset):
        
        # Set up vectors for coordinates
        x = self.x.copy()
        y = self.y.copy()
        
        # Loop through each region
        size = 0
        for i, boundary in enumerate(self.boundaries):
            looped_boundary = np.vstack((boundary, boundary[0]))
            
            # Get the number of turbines in that region
            n_turbs = int(n_sub_turbs[i])
            
            # Compute the curvilinear length of the boundary for that region.
            dimensional_distance = np.cumsum(np.sqrt( np.ediff1d(looped_boundary[:, 0], to_begin=0)**2 + np.ediff1d(looped_boundary[:, 1], to_begin=0)**2 ))
            
            # Compute the parametric curve, normalized from 0 to 1.
            distance = dimensional_distance / dimensional_distance[-1]

            # Set up interpolation functions along that parametric curve
            fx, fy = interp1d(distance, looped_boundary[:, 0]), interp1d(distance, looped_boundary[:, 1])
            
            polygon_area = self.polygons[i].area
            perimeter = dimensional_distance[-1]
            turb_per_perim = self.D * n_turbs / perimeter
            
            # Heurestic for this specific geometry to get some interior points
            magic_number = 1.0 - 2 * (turb_per_perim - coeff[i])
            
            if magic_number > 1:
                magic_number = 1
                
            n_boundary_turbs = int(magic_number * n_turbs)
            n_interior_turbs = n_turbs - n_boundary_turbs
            # print(f'interior: {n_interior_turbs}, boundary: {n_boundary_turbs}')
            
            # Create equidistant points in the parametric space with a random
            # starting point
            point_spacings = np.linspace(0, 1, n_boundary_turbs, endpoint=False)
            point_spacings += np.random.random()
            point_spacings[point_spacings > 1] -= 1
            
            # Compute the x and y coords for the turbines based off the
            # parametric equidistant points
            x[size:size+n_boundary_turbs] = fx(point_spacings)
            y[size:size+n_boundary_turbs] = fy(point_spacings)
            
            if n_interior_turbs > 0:
                new_x, new_y = self.place_turbine_on_inner_bound(n_interior_turbs, self.polygons[i], offset[i])
                x[size+n_boundary_turbs:size+n_boundary_turbs+n_interior_turbs] = new_x
                y[size+n_boundary_turbs:size+n_boundary_turbs+n_interior_turbs] = new_y
            
            # Keep track of the total number of turbines that have been placed
            size += n_turbs
            
        # Normalized the coordinates from physical to scaled coordinates
        self.x = self._norm(x, self.bndx_min, self.bndx_max)
        self.y = self._norm(y, self.bndy_min, self.bndy_max)
        
        self.x0, self.y0 = self.x.copy(), self.y.copy()
        
    def place_turbines_from_smart_starts(self, locsx, locsy):
        # Normalized the coordinates from physical to scaled coordinates
        self.x = self._norm(locsx, self.bndx_min, self.bndx_max)
        self.y = self._norm(locsy, self.bndy_min, self.bndy_max)
        
        self.x0, self.y0 = self.x.copy(), self.y.copy()

    def space_constraint(self, locs, rho=500):
        x = self._unnorm(locs[:, 0], self.bndx_min, self.bndx_max)
        y = self._unnorm(locs[:, 1], self.bndy_min, self.bndy_max)
                
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
        x = self._unnorm(locs[:, 0], self.bndx_min, self.bndx_max)
        y = self._unnorm(locs[:, 1], self.bndy_min, self.bndy_max)
        
        locs = np.vstack((x, y)).T
        points = MultiPoint(locs)
        dist_out = np.zeros((len(x), len(self.polygons)))
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
        
        return g  # KS_constraint


    def plot_layout_opt_results(self, sol=None, filename=None):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        if sol is not None:
            locsx = sol.getDVs()['x']
            locsy = sol.getDVs()['y']
            
            locsx = self._unnorm(locsx, self.bndx_min, self.bndx_max)
            locsy = self._unnorm(locsy, self.bndy_min, self.bndy_max)

        plt.figure(figsize=(9, 6))
        fontsize= 16
        
        x_init = self._unnorm(self.x0, self.bndx_min, self.bndx_max)
        y_init = self._unnorm(self.y0, self.bndy_min, self.bndy_max)
        plt.plot(x_init, y_init, 'ob')
        patches = []
        for coords in zip(x_init, y_init):
            patches.append(mpatches.Circle(coords, radius=self.D, linewidth=0.))
        collection = PatchCollection(patches, facecolor='b', alpha=0.2)
        plt.gca().add_collection(collection)
        
        if sol is not None:
            plt.plot(locsx, locsy, 'or')
            patches = []
            for coords in zip(locsx, locsy):
                patches.append(mpatches.Circle(coords, radius=self.D, linewidth=0.))
            collection = PatchCollection(patches, facecolor='r', alpha=0.2)
            plt.gca().add_collection(collection)
        
        plt.title(f'Initial AEP: {self.AEP_initial:.0f}, Optimized AEP: {self.get_AEP():.0f}', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        if sol is not None:
            plt.legend(['Old locations', 'New locations'], loc='lower center', \
                bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=fontsize)
        else:
            plt.legend(['Old locations'], loc='lower center', \
                bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=fontsize)
                
        for polygon in self.polygons:    
            xs, ys = polygon.exterior.xy    
            plt.plot(xs, ys, alpha=0.5, color='b')
            
        plt.tight_layout()
        
        if filename is not None:
            plt.savefig(filename)
            plt.close()
        else:
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