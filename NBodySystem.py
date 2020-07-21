"""
 -N-Body Solver & 3D orbit plotter / animator
 -A more generalized solution of the code found here:
 https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767

 - New features:
    - animation code includes
    - generalized n-body solver
    - class-object based code

 """

import scipy as sci
import scipy.integrate
import PlotTools as pt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D             # Do not remove this, it is used
import numpy as np
from itertools import combinations as cmb
import pandas as pd
import time
# import ffmpeg


# decorator to calculate duration
def timer_wrapper(func):
    # added arguments inside the inner1,
    # if function takes any arguments,
    # can be added like this.
    def inner1(*args, **kwargs):
        # storing time before function execution
        begin = time.time()
        ret_val = func(*args, **kwargs)
        end = time.time()
        print('\tExecution time for "%s()" is %5.3f seconds ' % (func.__name__, end - begin))
        return ret_val

    return inner1


def calc_dvdt_component(k1, delta_r, m2):
    """
    Calculate a single component of dv/dt calculation on a gravitational body
    :param k1: constant K1 calculated from non-denominational units
    :param m2:
    :param r1:
    :param r2:
    :return: dvdt component between body1 and body2
    """
    return k1 * m2 * delta_r / np.linalg.norm(delta_r) ** 3


def compute_com_component(r_data, m_data):
    """
    Compute the center of mass along a single cardinal direction
    :param r_data: tuple of radius values (r1, r2, r3) where rn = (x, y ,z)
    :param m_data: mass data for each body
    :return:
    """
    # r_data and m_data are zipped into radius, mass pairs based on accociated body
    return sum([m * r for m, r in list(zip(r_data, m_data))]) / sum(m_data)


def compute_com_pairs(bodies):

    com = {}
    masses = [b.m for b in bodies]
    r_sol = [list(b.r_sol.values()) for b in bodies]

    # Compute index combinations for each level in {n-bodies, n-bodies-1, ..., 2}
    for n in np.arange(len(bodies), 1, -1):
        index_cmbs = list(cmb(np.arange(0, len(bodies)), n))
        # body_combinations = list(cmb(bodies, n))
        # for comb in index_cmbs:
        for comb in index_cmbs:
            # Reshape from:
            # (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)  to
            # (x1, x2, x3 ...), (y1, y2, y3 ...), (z1, z2, z3 ...)
            com[comb] = [
                            [compute_com_component(tuple([r[c] for c in comb]), tuple([masses[c] for c in comb]))
                            for r in list(zip(*soln))]
                            for soln in list(zip(*r_sol))
                        ]
        if n <= 2:
            # do not compute com for 1 body
            break

    return com


def min_max_dimension(i, data_list):
    """ i = 0 corresponds to x, 1 to y, 2 to z"""
    return [func([func(values[i]) for values in data_list]) for func in [min, max]]


# ------------< Classes >-----------
class NBodySystem:

    def __init__(self, bodies, nd_units, dt, periods):
        """

        :param bodies:
        :param nd_units:
        :param dt:
        :param periods:
        """

        self.bodies = bodies
        self.G = 6.67408e-11
        self.nd = nd_units

        # K1 & K2 derived from nd_units
        self.K1 = self.G * self.nd['t'] * self.nd['m'] / (self.nd['r'] ** 2 * self.nd['v'])
        self.K2 = self.nd['v'] * self.nd['t'] / self.nd['r']

        # The body object should already have an r0 & v0 value
        self.inits = np.array([b.r0 for b in self.bodies] + [b.v0 for b in self.bodies]).flatten()

        self.dt = dt
        self.periods = periods
        self.time_steps = np.arange(0, periods, dt)

        # Used as storage for current value during solve
        self.drdt = []
        self.dvdt = []


        self.n_bodies = len(self.bodies)
        self.body_indexes = np.arange(0, self.n_bodies)
        self.r_com = None                                         # System center of mass, to be computed from solutions
        self.v_com = None
        dt_inverse = int(1/self.dt)
        print('%5d per period \tK1 = %3.3f , K2 = %3.3f ' % (dt_inverse, self.K1, self.K2))

    def execute(self):
        " Solve and update the bodies with the solutions"

        print('Solving \tΔt = %f s' % self.dt)
        r_sol, v_sol = self.solve()
        self.update_bodies(r_sol, v_sol, self.time_steps)
        self.compute_center_of_mass()
        self.compute_relative_positions()
        return

    @timer_wrapper
    def solve(self):
        """
        :return: position & velocity solution for each timestep of integration routine
        """
        n_body_sol = np.array(sci.integrate.odeint(self.n_body_eqns, self.inits, self.time_steps, args=())).T
        r_sol, v_sol = np.split(n_body_sol, 2)
        return np.split(r_sol, self.n_bodies), np.split(v_sol, self.n_bodies)

    def n_body_eqns(self, w, t):
        """
        :param w: flattened length n_bodies * n_directions (ie x,y,z) * 2 (position and velocity solutions)
        :param t: This is used by ode-int
        :return: dr/dt & dv/dt flattened solutions
                 shape: flattened length n_bodies * n_directions (ie x,y,z) * 2 (position and velocity solutions)
        """
        # Break flattened data into radius and velocity data for each body
        r_in, v_in = np.split(np.array(np.split(w, len(w)//3)), 2)

        self.dvdt = []
        self.drdt = np.array([self.K2 * v for v in v_in]).flatten()

        for i1 in self.body_indexes:
            # Calculate components for body i1 and sum as vectors rel. to each body (i2 bodies)
            other_bodies = (i for i in self.body_indexes if i != i1)
            self.dvdt.append(sum([calc_dvdt_component(self.K1, r_in[i2] - r_in[i1], self.bodies[i2].m) for i2 in tuple(other_bodies)]))

        # Return concatenated derivatives
        return np.concatenate([self.drdt,  np.array(self.dvdt).flatten()])

    def update_bodies(self, r_solns=None, v_solns=None, time_span=None):

        for b, body in enumerate(self.bodies):
            body.update_solutions(r_solns[b], v_solns[b], time_span)

    def reset_bodies(self):
        """
        Reset all x and y solutions for each body in the system
        :return:
        """
        for body in self.bodies:
            body.reset()

    @timer_wrapper
    def save_solutions(self, save=False, dir_='', tag=None):
        if save:
            print('\tSaving to %s' % dir_)
            for body in self.bodies:
                body.save_solution_df(tag=tag, dir_=dir_)

    def compute_center_of_mass(self):
        """
        Compute system center of mass.
        Data is reshaped from (x1, y1, z1), (x2, y2, z2) to (x1, x2, x3) (y1, y2, y3)
        :return:
        """
        masses = [b.m for b in self.bodies]
        r_sol = [list(b.r_sol.values()) for b in self.bodies]
        # Note: r is an (x1, x2,..., xn) tuple, same for y and z directions
        self.r_com = [[compute_com_component(r, masses) for r in list(zip(*soln))] for soln in list(zip(*r_sol))]
        self.r_com = {d: v for d, v in list(zip(['x', 'y', 'z'], self.r_com))}

    def compute_relative_positions(self):

        for body in self.bodies:
            r_com = np.array(list(body.r_sol.values())) - np.array(list(self.r_com.values()))
            body.r_com = {d: v for d, v in list(zip(['x', 'y', 'z'], r_com))}

    def to_df(self):
        """
        Put the position and velocity position data in to a Pandas DataFrame object
        :return: dict of dataframes for each NamedBody
        """

        dfs = {}
        for body in self.bodies:
            body_data = dict(list(body.r_sol.items()) + list(body.v_sol.items()))
            body_data['t'] = body.t
            df = pd.DataFrame(body_data)
            df = df.set_index('t')
            dfs[body.name] = df

        return dfs

    def plot_system(self, ax, key, *args, **kwargs):
        print(key)
        # if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, body in enumerate(self.bodies):
            kwargs['c'] = colors[i]
            ax = body.plot(ax, key, *args, **kwargs)
        ax.plot(*self.r_com.values(), c='k')
        ax.legend(labels=[body.name for body in self.bodies])

        return ax

    def animate_solution(self, key):
        """
        Credit for this function needs to be properly attributed, most of the code required to animate
        we directly copy and pasted - M. Bremner
        """

        fig = plt.figure(figsize=(12, 9))
        fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
        ax = fig.add_subplot(111, projection="3d")
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                  '#e377c2', '#d62728']

        data = {body.name: body.__getattribute__(key) for body in self.bodies}
        directions = list(list(data.values())[0].keys())

        # Plot initial values
        for i, (name, soln) in enumerate(data.items()):
            ax.scatter(*[s[0] for s in soln.values()], color=colors[i], marker="x", s=100)

        # Most recent points
        dots = {name: ax.scatter(*[s[0] for s in soln.values()], color=colors[i], marker="o", s=80)
                for i, (name, soln) in enumerate(data.items())}

        # Initialize plots
        anim_data = {key: ([], [], []) for key in data.keys()}
        lines = {key: ax.plot(*anim_data[key], lw=2.5, label=key) for key in anim_data.keys()}

        def n_body_init():
            """
                Init function for 3D animation of n-body problem
                Initialize axis parameters here
            """
            # Establish limits
            # x_lims, y_lims, z_lims = [pt.min_max_dimension(dim, data.values()) for dim in range(3)]
            # ax.set_xlim3d(*x_lims)
            # ax.set_ylim3d(*y_lims)
            # ax.set_zlim3d(*z_lims)

            ax.set_xlabel("Z", fontsize=14)
            ax.set_ylabel("Y", fontsize=14)
            ax.set_zlabel("X", fontsize=14)
            ax.set_title("Visualization of a(n) %d-body system for Δt = %3.3f \n" % (len(data), self.dt), fontsize=14)
            ax.legend(loc="upper left", fontsize=14)
            # return lines

        def n_body_update(frame):
            """
                Update function for 3D animation of n-body problem
                Update all data
            """
            frame = int(frame)
            # Iterate bodies & Append New Data
            for i, (name, item) in enumerate(anim_data.items()):
                # Iterate x, y, z directions
                for j, direction_data in enumerate(item):
                    dir_ = directions[j]
                    direction_data.append(data[name][dir_][frame])

            # Use this to update lims on the fly
            x_lims, y_lims, z_lims = [pt.min_max_dimension(dim, anim_data.values()) for dim in range(3)]
            ax.set_xlim3d(*x_lims)
            ax.set_ylim3d(*y_lims)
            ax.set_zlim3d(*z_lims)

            # Update Plot Data
            for name, line in lines.items():
                x, y, z = anim_data[name]
                line[0].set_data(x, y)
                line[0].set_3d_properties(z)
            # Update Lead Dot with most recent values
            for name, dot, in dots.items():
                dot_val = [[anim_data[name][i][-1]] for i in range(len(anim_data[name]))]
                dot = dots[name]
                dot._offsets3d = dot_val

        # Number of frames
        s = len(list(list(data.values())[0].values())[0])
        frame_list = np.linspace(0, s - 1, s)
        ani = FuncAnimation(fig, n_body_update, repeat=False, frames=frame_list, interval=10,
                            init_func=n_body_init, blit=False)

        plt.show()
        pass

    def __repr__(self):
        return str(self.n_bodies) + ' ' + str(self.dt) + '\t' + ', '.join([b.name for b in self.bodies])


class SystemCollection:
    """
    A collection of n-body systems
    """

    pass


class SystemPair(SystemCollection):
    """
    A collection of n-body systems with two systems
    """
    pass

    def __init__(self):

        self.shorter = None
        self.longer = None


