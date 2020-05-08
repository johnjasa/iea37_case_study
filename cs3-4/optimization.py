from jax import grad
import jax.numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
import iea37_aepcalc_test as ieatools
import copy

class Optimization():

    def __init__(self, file_name):
        self.file_name = file_name

        self.coords, self.fname_turb, self.fname_wr = \
            ieatools.getTurbLocYAML(self.file_name)
        self.cut_in, self.cut_out, self.rated_ws, self.rated_pow, self.D = \
            ieatools.getTurbAtrbtYAML(self.fname_turb)
        self.wd, self.wd_freq, self.ws, self.ws_freq, self.ws_bin_num, self.ws_min, self.ws_max = \
            ieatools.getWindRoseYAML(self.fname_wr)

    def _norm(self, val, x1, x2):
        return (val - x1)/(x2 - x1)
    
    def _unnorm(self, val, x1, x2):
        return np.array(val)*(x2 - x1) + x1

    @property
    def nturbs(self):
        """
        This property returns the number of turbines.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = np.shape(self.coords)[0]
        return self._nturbs

class cs3Opt(Optimization):
    def __init__(self, file_name, bnds_file_name, min_dist=None, 
                                                  jac=False, 
                                                  opt_method='SLSQP', opt_options=None):
        super().__init__(file_name)

        self.opt_method = opt_method
        if opt_options is None:
            self.opt_options = {'maxiter': 20, 'disp': True, \
                         'iprint': 2, 'ftol': 1e-7}
        else:
            self.opt_options = opt_options

        if jac is True:
            self.jac = grad(self._get_AEP_opt)
        else:
            self.jac = False

        self.boundaries = ieatools.getBoundaryAtrbtYAML(bnds_file_name)
        self.bndx_min = np.min([val[0] for val in self.boundaries])
        self.bndy_min = np.min([val[1] for val in self.boundaries])
        self.bndx_max = np.max([val[0] for val in self.boundaries])
        self.bndy_max = np.max([val[1] for val in self.boundaries])
        self.boundaries_norm = [[self._norm(val[0], self.bndx_min, \
                                self.bndx_max), self._norm(val[1], \
                                self.bndy_min, self.bndy_max)] \
                                for val in self.boundaries]

        self.AEP_initial = self.get_AEP()
        # self.x0 = self._coords_to_locs()
        self.set_initial_locs(self._coords_to_locs())

        if min_dist is not None:
            self.min_dist = min_dist
        else:
            self.min_dist = 2*self.D

        self._set_opt_bounds()
        self._generate_constraints()

    def set_grad(self, grad_func):
        self.grad = grad_func

    def set_initial_locs(self, locs):
        self.x0 = [self._norm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0:self.nturbs]] \
                + [self._norm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[self.nturbs:2*self.nturbs]]

    def get_AEP(self):
        AEP = ieatools.calcAEPcs3(self.coords, self.wd_freq, self.ws,
            self.ws_freq, self.wd, self.D, self.cut_in, self.cut_out, 
            self.rated_ws, self.rated_pow)

        # return np.sum(AEP)
        return AEP

    def _locs_to_coords(self, locs):
        self.coords = np.array(list(zip(locs[0:self.nturbs], locs[self.nturbs:2*self.nturbs])))

    def _coords_to_locs(self):
        locs = np.concatenate((self.coords[:,0], self.coords[:,1]))
        return locs

    def _get_AEP_opt(self, locs):
        locs_tmp = [self._unnorm(locx, self.bndx_min, self.bndx_max) for \
                locx in locs[0:self.nturbs]] \
                + [self._unnorm(locy, self.bndy_min, self.bndy_max) for \
                locy in locs[self.nturbs:2*self.nturbs]]
        self._locs_to_coords(np.array(locs_tmp))
        # print(locs_tmp)
        AEP = self.get_AEP()

        return -1 * AEP/self.AEP_initial

    def grad_get_AEP(self):
        pass

    def optimize(self):
        """
        Find optimized layout of wind turbines for power production given
        fixed atmospheric conditions (wind speed, direction, etc.).
        
        Returns:
            opt_locs (iterable): optimized locations of each turbine.
        """
        print('=====================================================')
        print('Optimizing turbine layout...')
        print('Number of parameters to optimize = ', len(self.x0))
        print('=====================================================')

        opt_locs_norm = self._optimize()

        print('Optimization complete!')

        opt_locs = [[self._unnorm(valx, self.bndx_min, self.bndx_max) \
            for valx in opt_locs_norm[0:self.nturbs]], \
            [self._unnorm(valy, self.bndy_min, self.bndy_max) \
            for valy in opt_locs_norm[self.nturbs:2*self.nturbs]]]

        return opt_locs
        
    def om_optimize(self):
        import openmdao.api as om
        from aep_comp import AEPComp
        from constraints_comp import ConstraintsComp
        
        prob = om.Problem()
        
        ivc = prob.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('locs', val=self.x0)
        
        prob.model.add_subsystem('turbines', AEPComp(opt_object=self), promotes=['*'])
        prob.model.add_subsystem('constraint_comp', ConstraintsComp(opt_object=self), promotes=['*'])
        
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.opt_settings['Major optimality tolerance'] = 1e-3
        
        prob.model.add_design_var('locs')
        prob.model.add_constraint('distance', upper=0.)
        prob.model.add_constraint('space', upper=0.)
        prob.model.add_objective('AEP')
        
        prob.setup()
        
        prob.run_driver()
        
        prob.model.list_inputs()
        prob.model.list_outputs()
        
        self.residual_plant = prob
        
        

    def _optimize(self):
        self.residual_plant = minimize(self._get_AEP_opt,
                                self.x0,
                                jac=self.grad,
                                method=self.opt_method,
                                bounds=self.bnds,
                                constraints=self.cons,
                                options=self.opt_options)

        opt_results = self.residual_plant.x
        
        return opt_results

    def _generate_constraints(self):
        # grad_constraint1 = grad(self._space_constraint)
        # grad_constraint2 = grad(self._distance_from_boundaries)

        tmp1 = {'type': 'ineq','fun' : lambda x,*args: \
                self._space_constraint(x, self.min_dist), \
                'args':(self.min_dist,)}
        tmp2 = {'type': 'ineq','fun' : lambda x,*args: \
                self._distance_from_boundaries(x, self.boundaries_norm), \
                'args':(self.boundaries_norm,)}

        self.cons = [tmp1, tmp2]

    def _set_opt_bounds(self):
        # self.bnds = [(0.0, 1.0) for _ in range(2*self.nturbs)]
        self.bnds = Bounds(0.0, 1.0, keep_feasible=True)

    def _space_constraint(self, x_in, min_dist, rho=50.):
        # x = np.nan_to_num(x_in[0:self.nturbs])
        # y = np.nan_to_num(x_in[self.nturbs:])

        x = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                     for valx in x_in[0:self.nturbs]]
        y = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                     for valy in x_in[self.nturbs:2*self.nturbs]]

        dist = [np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2) \
                for i in range(self.nturbs) \
                for j in range(self.nturbs) if i != j]
          
        g = 1 - np.array(dist) / min_dist
        
        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)
        
        return KS_constraint

    def _distance_from_boundaries(self, x_in, boundaries, rho=500.):  
        x = x_in[0:self.nturbs]
        y = x_in[self.nturbs:2*self.nturbs]
            
        dist_out = []

        for k in range(self.nturbs):
            dist = []
            in_poly = self._point_inside_polygon(x[k],y[k],boundaries)

            for i in range(len(boundaries)):
                boundaries = np.array(boundaries)
                p1 = boundaries[i]
                if i == len(boundaries)-1:
                    p2 = boundaries[0]
                else:
                    p2 = boundaries[i+1]

                px = p2[0] - p1[0]
                py = p2[1] - p1[1] 
                norm = px*px + py*py

                u = ((x[k] - boundaries[i][0])*px + \
                     (y[k] - boundaries[i][1])*py)/float(norm)

                if u <= 0:
                    xx = p1[0]
                    yy = p1[1]
                elif u >=1:
                    xx = p2[0]
                    yy = p2[1]
                else:
                    xx = p1[0] + u*px
                    yy = p1[1] + u*py

                dx = x[k] - xx
                dy = y[k] - yy
                dist.append(np.sqrt(dx*dx + dy*dy))

            dist = np.array(dist)
            if in_poly:
                dist_out.append(np.min(dist))
            else:
                dist_out.append(-np.min(dist))

        dist_out = np.array(dist_out)
        
        print(dist_out)
        
        g = - dist_out
        
        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)
        
        print(KS_constraint)
        print()

        return KS_constraint

    def _point_inside_polygon(self, x, y, poly):
        n = len(poly)
        inside =False

        p1x,p1y = poly[0]
        for i in range(n+1):
            p2x,p2y = poly[i % n]
            if y > min(p1y,p2y):
                if y <= max(p1y,p2y):
                    if x <= max(p1x,p2x):
                        if p1y != p2y:
                            xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x,p1y = p2x,p2y

        return inside

    def plot_layout_opt_results(self):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx_old = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                     for valx in self.x0[0:self.nturbs]]
        locsy_old = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                     for valy in self.x0[self.nturbs:2*self.nturbs]]
        locsx = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                 for valx in self.residual_plant.x[0:self.nturbs]]
        locsy = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                 for valy in self.residual_plant.x[self.nturbs:2*self.nturbs]]

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(locsx_old, locsy_old, 'ob')
        plt.plot(locsx, locsy, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts)-1:
                plt.plot([verts[i][0], verts[0][0]], \
                         [verts[i][1], verts[0][1]], 'b')        
            else:
                plt.plot([verts[i][0], verts[i+1][0]], \
                         [verts[i][1], verts[i+1][1]], 'b')
                         
    def om_plot_layout_opt_results(self):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        locsx_old = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                     for valx in self.x0[0:self.nturbs]]
        locsy_old = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                     for valy in self.x0[self.nturbs:2*self.nturbs]]
        locsx = [self._unnorm(valx, self.bndx_min, self.bndx_max) \
                 for valx in self.residual_plant['locs'][0:self.nturbs]]
        locsy = [self._unnorm(valy, self.bndy_min, self.bndy_max) \
                 for valy in self.residual_plant['locs'][self.nturbs:2*self.nturbs]]

        plt.figure(figsize=(9,6))
        fontsize= 16
        plt.plot(locsx_old, locsy_old, 'ob')
        plt.plot(locsx, locsy, 'or')
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel('x (m)', fontsize=fontsize)
        plt.ylabel('y (m)', fontsize=fontsize)
        plt.axis('equal')
        plt.grid()
        plt.tick_params(which='both', labelsize=fontsize)
        plt.legend(['Old locations', 'New locations'], loc='lower center', \
            bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=fontsize)

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts)-1:
                plt.plot([verts[i][0], verts[0][0]], \
                         [verts[i][1], verts[0][1]], 'b')        
            else:
                plt.plot([verts[i][0], verts[i+1][0]], \
                         [verts[i][1], verts[i+1][1]], 'b')