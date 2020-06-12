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
import helperFunctions as hf

nd = {
    'm': 1.989e+30,  # kg, mass of the sun
    'r': 5.326e+12,  # dist btwn. Alpha Centauri A & B
    'v': 30000,  # m/s v-rel of earth around the sun
    't': 79.91 * 365 * 24 * 3600 * 0.51  # orbital period of Alpha Centauri
}


# ----------< Classes >---------
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

    def execute(self):
        " Solve and update the bodies with the solutions"
        r_sol, v_sol = self.solve()
        self.update_bodies(r_sol, v_sol, self.time_steps)
        return

    # @hf.calculate_time
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
            self.dvdt.append(sum([calc_dvdt_component(self.K1, r_in[i1], r_in[i2], self.bodies[i2].m) for i2 in tuple(other_bodies)]))

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

    def save_solutions(self, save=False, tag=None):
        if save:
            for body in self.bodies:
                body.save_solution(tag)

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

    def save_solution(self, tag=None):
        if tag == None:
            tag = ''


        df = pd.DataFrame(self.r_sol)
        df['t'] = self.t
        # print('wtf')
        # print('wtf')
        # df['norm'] = df['x', 'y', 'z'].apply(np.linalg.norm)
        df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        df.to_csv('results//' + self.name + '-r_sol-' + tag + '.csv', index=False)
        df = pd.DataFrame(self.v_sol)
        df['t'] = self.t
        # df['norm'] = df[['x', 'y', 'z']].apply(np.linalg.norm)
        df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        df.to_csv('results//' + self.name + '-v_sol'  + tag +  '.csv', index=False)


def UC_time_evaluation(body_list, nd_values):
    # Example:
    # 1. EVALUATE EXECUTION TIME >-----
    # UC_time_evaluation([two_bodies, three_bodies, four_bodies, five_bodies], nd)
    periods, steps_per_prd = 50, 200
    timespan = np.linspace(0, periods - 1, periods * steps_per_prd)

    for bodies in body_list:
        print('\n%d Bodies, # of pts: %d' % (len(bodies), len(timespan)))
        nbs = NBodySystem(bodies, nd_values)
        nbs.execute(nbs, timespan)
        nbs.reset_bodies()


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


def calc_dvdt_component(k1, r1, r2, m2):
    """
    Calculate a single component of dv/dt calculation on a gravitational body
    :param k1: constant K1 calculated from non-denominational units
    :param m2:
    :param r1:
    :param r2:
    :return: dvdt component between body1 and body2
    """
    return k1 * m2 * (r2 - r1) / np.linalg.norm(r2-r1) ** 3


def unanimated_orbit_plot(ax, data_pos, data_com=None):
    """

    :param r_a: x, y, z data for body A
    :param r_b: x, y, z data for body B
    :param r_com: x, y, z data for Center of Mass
    :return:
    """

    # Create 3D axes

    # ax = fig.add_subplot(121, projection="3d")
    # ax2 = fig.add_subplot(122)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62727', '#282828', 'blue']

    # Plot Lines



    for i, (name, soln) in enumerate(data_pos.items()):
        print(soln)
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


def animate_solution(fig, data_pos, data_com):

    # Create 3D axes
    # fig = plt.figure(figsize=(12, 9))
    # fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
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

def main():

    # Initialize Bodies
    alpha_centauri_a = NamedBody('Alpha Centauri A', 1.1, np.array([-0.5, 0, 0], dtype="float64"), np.array([0.01, 0.01, 0], dtype="float64"))
    alpha_centauri_b = NamedBody('Alpha Centauri B', 0.907, np.array([0.5, 0, 0], dtype="float64"), np.array([-0.05, 0, -0.1], dtype="float64"))
    body_c = NamedBody('Body C', 1.035, [0, 1, 0], [0, -0.01, 0])

    two_bodies = [alpha_centauri_a, alpha_centauri_b]
    three_bodies = two_bodies + [body_c]


    input_bodies = two_bodies
    p, timesteps = 20, [0.1, 0.01]
    # Initialize each system
    Systems = [NBodySystem(copy.deepcopy(input_bodies), nd, t_step=step, periods=p)  for step in timesteps]
    # Solve each system
    for system in Systems:
        system.execute()
    # Each system has a set of bodies which store the solutions
    position_data = [{b.name: list(b.r_sol.values()) for b in S.bodies} for S in Systems]

    # Plot the data
    fig = plt.figure(figsize=(14, 8))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})

    # 1. Animate one the solution
    animate_solution(fig, data_pos=position_data[1], data_com=None)
    #
    # 2. 3D plot several solutions on separate
    # ax1 = fig.add_subplot(121, projection="3d")
    # ax1 = unanimated_orbit_plot(ax1, position_data[0], data_com=None)
    # ax1 = fig.add_subplot(122, projection="3d")
    # ax2 = unanimated_orbit_plot(ax1, position_data[1], data_com=None)
    plt.show()
    exit()


if __name__ == "__main__":
    main()

