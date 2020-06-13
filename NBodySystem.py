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
from PIL import ImageColor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial import distance_matrix
from itertools import combinations as cmb
import pandas as pd
import copy
import io
import os

nd = {
    'm': 1.989e+30,  # kg, mass of the sun
    'r': 5.326e+12,  # dist btwn. Alpha Centauri A & B
    'v': 30000,  # m/s v-rel of earth around the sun
    't': 79.91 * 365 * 24 * 3600 * 0.51  # orbital period of Alpha Centauri
}

import helperFunctions as hf

# -----------< Classes >----------
class NBodySystem:

    def __init__(self, bodies, nd_units, t_step, periods):

        self.bodies = bodies

        self.G = 6.67408e-11
        self.nd = nd_units
        # K1 & K2 derived from nd_units
        self.K1 = self.G * self.nd['t'] * self.nd['m'] / (self.nd['r'] ** 2 * self.nd['v'])
        self.K2 = self.nd['v'] * self.nd['t'] / self.nd['r']
        self.inits = np.array([b.r0 for b in self.bodies] + [b.v0 for b in self.bodies]).flatten()

        self.delta_t = t_step
        self.time_steps = np.arange(0, periods, t_step)
        self.drdt = []
        self.dvdt = []

        self.n_bodies = len(self.bodies)
        self.body_indexes = np.arange(0, self.n_bodies)
        self.com = None                                         # System center of mass, to be computed from solutions



    def execute(self):
        " Solve and update the bodies with the solutions"
        print('\tΔt = %f s' % self.delta_t)
        r_sol, v_sol = self.solve()
        self.update_bodies(r_sol, v_sol, self.time_steps)
        return

    @hf.calculate_time
    def solve(self):
        """
        :param time_span: lin-space of time slices
        :return:
        """
        n_body_sol = np.array(sci.integrate.odeint(self.n_body_eqns, self.inits, self.time_steps, args=())).T
        r_sol, v_sol = np.split(n_body_sol, 2)
        return np.split(r_sol, self.n_bodies), np.split(v_sol, self.n_bodies)

    def n_body_eqns(self, w, t):
        """
        :param w: flattened length n_bodies * n_directions (ie x,y,z) * 2 (position and velocity solutions)
        :param t: Nothing done with this
        :param G:
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

    @hf.calculate_time
    def save_solutions(self, save=False, dir_='', tag=None):
        if save:
            for body in self.bodies:
                body.save_solution(tag=tag, dir_=dir_)

    def compute_center_of_mass(self):
        masses = [b.m for b in self.bodies]
        r_sol = [list(b.r_sol.values()) for b in self.bodies]
        self.com = [[compute_com(r, masses) for r in list(zip(*soln))] for soln in list(zip(*r_sol))]

    def compute_relative_positions(self):
        for body in self.bodies:
            body.r_com = np.array(list(body.r_sol.values())) - np.array(self.com)


    def __repr__(self):
        return str(self.n_bodies) + ' ' + str(self.delta_t) + '\t' + ', '.join([b.name for b in self.bodies])


class GalacticBody:

    def __init__(self, m, r0, v0):

        self.m = m
        self.r0 = r0        # initial position
        self.v0 = v0        # initial velocity

        self.r = []         # current position
        self.v = []         # current velocity

        self.r_sol = {'x': [], 'y': [], 'z': []}
        self.v_sol = {'x': [], 'y': [], 'z': []}

        self.r_com = {'x': [], 'y': [], 'z': []}

        self.t = []

    def __repr__(self):
        return '\t'.join([str(self.m), str(self.r0), str(self.v0)])

    def reset(self):
        self.r_sol = {'x': [], 'y': [], 'z': []}
        self.v_sol = {'x': [], 'y': [], 'z': []}

    def update_solutions(self,r_solns=None, v_solns=None, t=None):

        self.t.extend(t)
        if r_solns is not None:
            for i, key in enumerate(self.r_sol.keys()):
                self.r_sol[key].extend(r_solns[i])

        if v_solns is not None:
            for i, key in enumerate(self.v_sol.keys()):
                self.v_sol[key].extend(v_solns[i])


class NamedBody(GalacticBody):

    def __init__(self, name, *args, **kwargs):
        super(NamedBody, self).__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return '\t'.join((self.name, super(NamedBody, self).__repr__()))

    def save_solution(self, tag=None, dir_=''):

        """

        :param tag: name of experiment (ex: tag = '-50-1000' if  50 periods 1000 pts per period)
        :return:
        """
        if tag == None:
            tag = ''

        d = {}
        d.update(self.r_sol)
        d['t'] = self.t
        d['norm'] = [np.linalg.norm(item) for item in list(zip(*(self.r_sol.values())))]
        p = os.path.join(dir_, self.name + '-r_sol' + tag + '.csv')
        hf.saveTupleList([tuple(d.keys())] + list(zip(*d.values())), p)

        d.update(self.v_sol)
        d['t'] = self.t
        d['norm'] = [np.linalg.norm(item) for item in list(zip(*(self.v_sol.values())))]
        p = os.path.join(dir_, self.name + '-v_sol' + tag + '.csv')
        hf.saveTupleList([tuple(d.keys())] + list(zip(*d.values())), p)

    def save_solution_df(self, tag=None):
        """
        Use pandas dataframes to compute norm and save to csv
        *note: this is much slower than the method save_solution, mostly because
        df.apply is much slower than working directly on the values
        :param tag:
        :return:
        """
        if tag == None:
            tag = ''
        df = pd.DataFrame(self.r_sol)
        df['t'] = self.t

        df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        df.to_csv('results//' + self.name + '-r_sol-' + tag + '.csv', index=False)
        df = pd.DataFrame(self.v_sol)
        df['t'] = self.t
        df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        df.to_csv('results//' + self.name + '-v_sol'  + tag +  '.csv', index=False)


# ----------< Functions >---------
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
                            [compute_com(tuple([r[c] for c in comb]), tuple([masses[c] for c in comb]))
                            for r in list(zip(*soln))]
                            for soln in list(zip(*r_sol))
                        ]
        if n <= 2:
            # do not compute com for 1 body
            break

    return com


def compute_com(r_data, m_data):
    """
    Compute the center of mass along a single cardinal direction
    :param r_data: tuple of radius values (r1, r2, r3) where rn = (x, y ,z)
    :param m_data: mass data for each body
    :return:
    """
    # r_data and m_data are zipped into radius, mass pairs based on accociated body
    return sum([m * r for m, r in list(zip(r_data, m_data))]) / sum(m_data)


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


def unanimated_orbit_plot(ax, data_pos, data_com=None):
    """
    :param ax: axis object to plot on
    :param data_pos: x, y, z data for each body
    :param r_com: x, y, z data for Center of Mass
    :return:
    """

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62727', '#282828', 'blue', 'k']

    # Plot Lines
    for i, (name, soln) in enumerate(data_pos.items()):
        ax.plot(*soln, lw=2.1, linestyle='solid')
        ax.scatter(*[s[-1] for s in soln], marker="o", s=100, label=name)
        ax.scatter(*[s[0] for s in soln], color=colors[i],  marker="x", s=100)

    # Plot Center of Mass Data
    if data_com is not None:
        for i, (name, paired_com) in enumerate(sorted(data_com.items(), key=lambda x: len(x[0]), reverse=True)):
            if i == 0:
                lw = 3.0
                s = 80
            else:
                lw = 1.5
                s = 25
            ax.plot(*paired_com, color="k", lw=lw, linestyle='dashed') #, label=str(name))
            ax.scatter(*[p[-1] for p in paired_com], color='k', marker="o", s=s)
            ax.scatter(*[p[0] for p in paired_com], color='k', marker="x", s=50)

    # Plot the final positions of the stars
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_zlabel("z", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a %d-body system\n" % len(data_pos.items()), fontsize=14)
    ax.legend(loc="lower left", fontsize=12)

    # pane_color = '#b5d3e7'
    # rgba = [i/256 for i in ImageColor.getrgb(pane_color)] + [0.99]
    # ax.xaxis.set_pane_color(rgba)
    # ax.yaxis.set_pane_color(rgba)
    # ax.zaxis.set_pane_color(rgba)
    return ax


def animate_solution(data_pos, data_com):

    # Create 3D axes
    fig = plt.figure(figsize=(12, 9))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    ax = fig.add_subplot(111, projection="3d")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']



    # Plot initial values
    [ax.scatter(*[s[0] for s in (soln)], color=colors[i], marker="x", s=100)
        for i, (name, soln) in enumerate(data_pos.items())]

    # Plot a dot for most recent value
    dots = {name: ax.scatter(*[s[0] for s in (soln)], color=colors[i], marker="o", s=80)
            for i, (name, soln) in enumerate(data_pos.items())}

    # Initialize plots
    anim_data = {key: ([], [], []) for key in data_pos.keys()}
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
        ax.set_title("Visualization of an n-body system\n", fontsize=14)
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
                direction_data.append(data_pos[name][j][frame])

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

    s = np.array([v for v in data_pos.values()]).shape[-1]

    frame_list = np.linspace(0, s-1, s)

    ani = FuncAnimation(fig, n_body_update, repeat=False, frames=frame_list, interval=1,
                        init_func=n_body_init, blit=False)

    plt.show()


def EXAMPLE_pos_vel_plot(pos_data, vel_data, dt):
    #. 2. Position Velocity Plot
    fig = plt.figure(figsize=(14, 8))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    fig.suptitle('Visualization of Two body System: delta-t: %4.4f' % dt)
    ax1 = fig.add_subplot(121, projection="3d")
    ax1 = unanimated_orbit_plot(ax1, pos_data, data_com = None)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2 = unanimated_orbit_plot(ax2, vel_data, data_com=None)
    ax1.set_title('Position relative to IRF')
    ax2.set_title('Velocity relative to IRF')
    plt.subplots_adjust(left=0.1, right=0.95, top=1, wspace=0.1, hspace=0.1)


def EXAMPLE_compare_timesteps(pos_data, vel_data, timesteps):

    # a. Position plots
    fig = plt.figure(figsize=(14, 8))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    fig.suptitle('Position of system at various time steps')
    data = pos_data
    ax1 = fig.add_subplot(121, projection="3d")
    ax1 = unanimated_orbit_plot(ax1, data[0], data_com=None)
    ax2 = fig.add_subplot(122, projection="3d")
    ax2 = unanimated_orbit_plot(ax2, data[1], data_com=None)
    ax1.set_title('Delta-t = %4.4f ' % timesteps[0])
    ax2.set_title('Delta-t = %4.4f' % timesteps[1])

    # b. Velocity Plots
    fig2 = plt.figure(figsize=(14, 8))
    fig2.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    fig2.suptitle('Velocity of system at various time steps')
    data = vel_data
    ax3 = fig2.add_subplot(121, projection="3d")
    ax3 = unanimated_orbit_plot(ax3, data[0], data_com=None)
    ax4 = fig2.add_subplot(122, projection="3d")
    ax4 = unanimated_orbit_plot(ax4, data[1], data_com=None)
    ax3.set_title('Delta-t = %f ' % timesteps[0])
    ax4.set_title('Delta-t = %f' % timesteps[1])
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.1, hspace=0.1)


def EXAMPLE_single_plot(data):
    #. 2. Position Velocity Plot
    fig = plt.figure(figsize=(14, 8))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    ax1 = fig.add_subplot(111, projection="3d")
    ax1 = unanimated_orbit_plot(ax1, data, data_com=None)


def main():

    # Initialize Bodies
    alpha_centauri_a = NamedBody('Alpha Centauri A', 1.1, np.array([-0.5, 0, 0], dtype="float64"), np.array([0.01, 0.01, 0], dtype="float64"))
    alpha_centauri_b = NamedBody('Alpha Centauri B', 0.907, np.array([0.5, 0, 0], dtype="float64"), np.array([-0.05, 0, -0.1], dtype="float64"))
    body_c = NamedBody('Body C', 1.0, [0, 1, 0], [0, -0.01, 0])

    two_bodies = [alpha_centauri_a, alpha_centauri_b]
    three_bodies = two_bodies + [body_c]

    # Initialize each system
    input_bodies = two_bodies
    p, timesteps = 50, [0.01, 0.1]

    # Make an n-body-system for each time-step
    Systems = [NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  for step in timesteps]


    # Solve each system w/ optional saving
    save_ = False
    for i, system in enumerate(Systems):
        print('%d. Solving' % i)
        system.execute()
        system.compute_center_of_mass()
        system.compute_relative_positions()
        print('\tSaving')
        system.save_solutions(save=save_, dir_='results//two-body', tag='-' + str(len(system.bodies)) + '-' + str(p) + '-' + str(int(1/timesteps[i])))
    print('Solving & Saving is complete')
    # Extract data from each system
    position_data = [{body.name: list(body.r_sol.values()) for body in S.bodies} for S in Systems]
    velocity_data = [{body.name: list(body.v_sol.values()) for body in S.bodies} for S in Systems]
    pos_relative_data = [{body.name: body.r_com for body in S.bodies} for S in Systems]
    error_data = []


    # 1. Plot a single set of solution data
    # EXAMPLE_single_plot(pos_relative_data[0])

    # Plot the data
    # 2. Animate one the solution
    # animate_solution(data_pos=position_data[0], data_com=None)

    # 3. Position velocity graph
    # EXAMPLE_pos_vel_plot(position_data[0], velocity_data[0], timesteps[0])

    # 4. Several Figures: Compare position and velocity for two distinct time_step values
    # EXAMPLE_compare_timesteps(position_data, velocity_data, timesteps)

    plt.show()
    print('Thank You')
    exit()


if __name__ == "__main__":
    main()

