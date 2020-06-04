""" Two body solver & 3D orbit plotter / animator"""

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

nd = {
    'm': 1.989e+30,  # kg, mass of the sun
    'r': 5.326e+12,  # dist btwn. Alpha Centauri A & B
    'v': 30000,  # m/s v-rel of earth around the sun
    't': 79.91 * 365 * 24 * 3600 * 0.51  # orbital period of Alpha Centauri
}


def compute_com(r_data, m_data):

    return (sum([m_data[i] * r_data[i] for i in range(len(r_data))])) / sum(m_data)


def calc_dvdt(K1, m2, r1, r2, r_rel):
    """ Calculate a single component of dv/dt calculation on a gravitational body """
    return K1 * m2 * (r2 - r1) / r_rel ** 3


def unanimated_orbit_plot(data, r_com=None):
    """

    :param r_a: x, y, z data for body A
    :param r_b: x, y, z data for body B
    :param r_com: x, y, z data for Center of Mass
    :return:
    """

    # Create 3D axes
    fig, _ = pt.tight_fig(1, 1)
    ax = fig.add_subplot(111, projection="3d")
    # dims = len(data[0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for i, (name, soln) in enumerate(data.items()):
        ax.plot(*soln, lw=1.7)
        ax.scatter(soln[0][-1], soln[1][-1], soln[2][-1], marker="o", s=100, label=name)

    for i, (name, soln) in enumerate(data.items()):
        ax.scatter(soln[0][0], soln[1][0], soln[2][0], color=colors[i],  marker="x", s=100)

    if r_com is not None:
        ax.plot(*r_com, color="k")

    # Plot the final positions of the stars
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)
    ax.set_zlabel("z", fontsize=14)
    ax.set_title("Visualization of orbits of stars in a %d-body system\n" % len(data), fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    rgba = [i/256 for i in ImageColor.getrgb("#b5d3e7")] + [0.5]
    # rgba = (0, 1, 0, 0.5)
    ax.xaxis.set_pane_color(rgba)
    ax.yaxis.set_pane_color(rgba)
    ax.zaxis.set_pane_color(rgba)

    return ax


def animated_orbit_plot(data):

    """ Data should be in this format:
        dict of name: data pairs
        data consists of position data for each cardinal direction (x, y ,z)"""

    # Create 3D axes
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # Plot initial values
    init_vals = [ax.scatter(*[soln[i][0] for i in range(len(soln))], color=colors[i], marker="x", s=100)
        for i, (name, soln) in enumerate(data.items())]

    dots = {name: ax.scatter(*[soln[i][0] for i in range(len(soln))], color=colors[i], marker="o", s=80)
        for i, (name, soln) in enumerate(data.items())}

    # Establish limits
    x_lims, y_lims, z_lims = [pt.min_max_dimension(dim, data.values()) for dim in range(3)]

    # Init plots
    anim_data = {key: ([], [], []) for key in data.keys()}
    lines = {key: ax.plot(*anim_data[key], lw=2.5, label=key) for key in anim_data.keys()}

    def n_body_init():
        """
            Init function for 3D animation of n-body problem
            Initialize axis parameters here
        """
        ax.set_xlim3d(*x_lims)
        ax.set_ylim3d(*y_lims)
        ax.set_zlim3d(*z_lims)

        ax.set_xlabel("x-coordinate", fontsize=14)
        ax.set_ylabel("y-coordinate", fontsize=14)
        ax.set_zlabel("z-coordinate", fontsize=14)
        ax.set_title("Visualization of orbits of stars in a two-body system\n", fontsize=14)
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
                direction_data.append(data[name][j][frame])

        # Update Plot Data
        for name, line in lines.items():
            x, y, z = anim_data[name]
            line[0].set_data(x, y)
            line[0].set_3d_properties(z)

        for name, dot, in dots.items():

            dot_val = [[anim_data[name][i][-1]] for i in range(len(anim_data.values()))]
            dot = dots[name]
            dot._offsets3d = dot_val


    s = np.array([v for v in data.values()]).shape[-1]
    frame_list = np.linspace(0, s-1, s)

    ani = FuncAnimation(fig, n_body_update, frames=frame_list, interval=1,
                        init_func=n_body_init, blit=False)
    plt.show()

# ----------< Classes >---------
class NBodySystem:

    def __init__(self, bodies, nd_units):

        self.bodies = bodies

        # compute initial dm
        A = [body.r0 for body in self.bodies]
        dm = distance_matrix(A, A)
        print(dm)
        # get update indexes
        dm_indexes = list(cmb([1, 2, 3], 2))
        self.G = 6.67408e-11
        self.nd = nd_units
        self.K1 = self.G * self.nd['t'] * self.nd['m'] / (self.nd['r'] ** 2 * self.nd['v'])
        self.K2 = self.nd['v'] * self.nd['t'] / self.nd['r']

        self.init_params = np.array([body.r0 for body in bodies] + [body.v0 for body in bodies]).flatten()
        print(self.init_params)
        # self.init_params = np.array([body_a.r0, body_b.r0, body_a.v0, body_b.v0]).flatten()

    def solve(self, time_span, inits):
        n_body_sol = np.array(sci.integrate.odeint(self.n_body_equations, inits, time_span,
                                                     args=(self.G,))).T
        r_sol = n_body_sol[:int(len(n_body_sol)/2)]
        v_sol = n_body_sol[int(len(n_body_sol)/2):]

        r_sols = [tuple(r_sol[3*n:(3*n + 3)]) for n in range(len(self.bodies))]
        v_sols = [tuple(v_sol[3*n:(3*n + 3)]) for n in range(len(self.bodies))]

        # r1_sol, r2_sol, r3_sol = tuple(n_body_sol[:3]), tuple(n_body_sol[3:6]), tuple(n_body_sol[6:9])
        # v1_sol, v2_sol, v3_sol = tuple(n_body_sol[9:12]), tuple(n_body_sol[12:15]), tuple(n_body_sol[15:18])
        # return r1_sol, r2_sol, r3_sol, v1_sol, v2_sol, v3_sol
        return r_sols, v_sols

    def n_body_equations(self, w, t, G):
        # Separate flattened data
        r = [w[n:n+3] for n in np.arange(0, 3*len(self.bodies), 3)]
        v = [w[n:n+3] for n in np.arange(3*len(r), 3*len(r) + 3*len(self.bodies), 3)]

        # Magnitude of relative radius for each planet combination
        r_rel = {comb: np.linalg.norm(r[comb[1]] - r[comb[0]]) for comb in list(cmb(list(range(len(self.bodies))), 2))}

        drdt = [self.K2 * v[i] for i in range(len(self.bodies))]
        dvdt = []
        body_indexes = np.arange(0, len(self.bodies))
        for i1 in body_indexes:
            other_bodies = ([i for i in body_indexes if i != i1])
            # dvdt.append(sum([self.calc_dvdt(r, r_rel, i1, i2) for i2 in other_bodies]))
            dvdt.append(sum([calc_dvdt(self.K1, self.bodies[i2].m, r[i1], r[i2], r_rel[min((i1, i2), (i2, i1))]) for i2 in other_bodies]))
        derivs = np.array(drdt).flatten(),  np.array(dvdt).flatten()
        return np.concatenate(derivs)


    def calc_dvdt(self, r, r_rel, i1, i2):
        return self.K1 * self.bodies[i2].m * (r[i2] - r[i1]) / r_rel[min((i1, i2), (i2, i1))] ** 3


class GalacticBody:

    def __init__(self, m, r0, v0):

        self.m = m
        self.r0 = r0
        self.v0 = v0


    def __repr__(self):
        return '\t'.join([str(self.m), str(self.r0), str(self.v0)])


class NamedBody(GalacticBody):

    def __init__(self, name, *args, **kwargs):
        super(NamedBody, self).__init__(*args, **kwargs)
        self.name = name




def main():
    # Idea: Calculate the relative error at each time-step given the step size
    alpha_centauri_a = NamedBody('Alpha Centauri A', 1.1, np.array([-0.5, 0, 0], dtype="float64"), np.array([0.01, 0.01, 0], dtype="float64"))
    alpha_centauri_b = NamedBody('Alpha Centauri B', 0.907, np.array([0.5, 0, 0], dtype="float64"), np.array([-0.05, 0, -0.1], dtype="float64"))
    body_c = NamedBody('Body C', 1.0, [0, 1, 0], [0, -0.01, 0])

    three_bodies = (alpha_centauri_a, alpha_centauri_b, body_c)
    two_bodies = (alpha_centauri_a, alpha_centauri_b)

    bodies = three_bodies
    NBS = NBodySystem(bodies, nd)
    masses = tuple([body.m for body in NBS.bodies])


    periods, steps_per_prd = 20, 200
    timespan = np.linspace(0, periods-1, periods * steps_per_prd)

    init_params = np.array([b.r0 for b in NBS.bodies] + [b.v0 for b in NBS.bodies]).flatten()
    # Solve
    r_sol, _ = NBS.solve(timespan, init_params)
    data_dict = dict(zip([body.name for body in NBS.bodies], r_sol))

    # Center of mass calcs
    r_solns = list(zip(*r_sol))
    r_com = [[compute_com(radii, masses) for radii in list(zip(*soln))] for soln in r_solns]

    print('Data Summary:')
    for key, vals in data_dict.items():
        print('\t' + key)
        for dir in vals:
            print('\t\t' + ', '.join(str(v) for v in dir[:10]))

    animated_orbit_plot(data=data_dict)
    # unanimated_orbit_plot(data_dict, r_com=r_com)
    plt.show()
    exit()

    # def compute_all_COM():
        # dm_indexes = lis, 2))

    # r_com13 = [[compute_com((r1, r3), (m1, m3)) for r1, _,  r3 in list(zip(*soln))] for soln in solutions]
    # r_com12 = [[compute_com((r1, r2), (m1, m2)) for r1, r2, _ in list(zip(*soln))] for soln in solutions]
    # r_com23 = [[compute_com((r2, r3), (m2, m3)) for _, r2, r3 in list(zip(*soln))] for soln in solutions]
    # ax.plot(*r_com13, color='k', linestyle='dashed')
    # ax.plot(*r_com12, color='k', linestyle='dashed')
    # ax.plot(*r_com23, color='k', linestyle='dashed')


if __name__ == "__main__":
    main()
