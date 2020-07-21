"""
 -N-Body Solver & 3D orbit plotter / animator
 -A more generalized solution of the code found here:
 https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767

 - New features:
    - animation code includes
    - generalized n-body solver
    - class-object based code
 """

from GalacticBody import NamedBody
from NBodySystem import NBodySystem
import decimal
import matplotlib.pyplot as plt
import numpy as np
import copy
import pandas as pd
from itertools import combinations as cmb
import ffmpeg

nd = {
    'm': 1.989e+30,  # kg, mass of the sun
    'r': 5.326e+12,  # meters, dist btwn. Alpha Centauri A & B
    # 'r': 5.326e+11,  # meters, dist btwn. Alpha Centauri A & B
    'v': 30000,  # m/s v-rel of earth around the sun
    't': 79.91 * 365 * 24 * 3600 * 0.51  # orbital period of Alpha Centauri
}

nd_units = {'m': 'Sun Masses',
            'r': ' Multiple of dist. btwn Alpha A / B',
            'v': 'Earth Velocity',
            't': 'Period of Alpha Centauri'}


# -----------< Functions >-----------

def compute_diff_between_systems(sys_a, sys_b):

        """
        Compute differences along each cardinal direction for two identical systems
        with different timesteps.

        * A note on rounding: You'll note round(10) in the code. The time steps are
            rounded to 10 places to manage floating point precision errors. Don't use a
            timestep < 10**-10 unless you increase 10 accordingly

        :param sys_a: NBodySystem obj.
        :param sys_b: NBodySystem obj. to be compared with sys_a
        :return:
        """

        print('Computing differences')

        # Shorter system has fewer points i.e. a larger dt value
        sys_short, sys_long = sorted([sys_a, sys_b], key=lambda x: x.dt, reverse=True)

        # Make a dictionary of Pandas DataFrames for each system  {body_name: df}
        df_dict_short, df_dict_long = sys_short.to_df(), sys_long.to_df()

        difference_dict = {}
        for name in df_dict_short.keys():
            if name in df_dict_long:
                df_short = df_dict_short[name]
                df_short.index = df_short.index.values.round(10)
                df_long = df_dict_long[name]
                df_long.index = df_long.index.values.round(10)
                df_long_new = df_long.loc[df_long.index.intersection(df_short.index)]
                df_delta = df_long_new.sub(df_short, fill_value=0, axis=0)
                difference_dict[name] = df_delta
            else:
                raise ValueError('Key %s should be in both solutions' % name)
        return difference_dict


def df_index_intersection(df_a=None, df_b=None):
    """
     Compute the intersection of common index values between two dataframes
    :return: all rows from the longer df which contain matching indexes with the shorter df
    """
    df_short = min([df_a, df_b], key=len)
    df_long = max([df_a, df_b], key=len)
    return df_long.loc[df_long.index.intersection(df_short.index)]


# -----------< Examples >------------
def plot_difference(sys_a, sys_b):
    """
    Plot the difference in position for two identical systems solved with different timesteps
    :param Systems: Two systems to compare
    :return:
    """


    # The function computer_diff-between_systems needs to be fixed so that
    # difference_dict[name] = df where cnames = [x, y, z, dx, dy, dz]
    difference_dict = compute_diff_between_systems(sys_a, sys_b)
    pos_keys, vel_keys = np.array(['x', 'y', 'z']), np.array(['dx', 'dy', 'dz'])
    plot_keys = pos_keys

    difference_dict = difference_dict['Alpha Centauri A']
    # df_c = difference_dict['Body C']

    min_, max_ = [func([func(difference_dict[key]) for key in plot_keys]) for func in [min, max]]



    # Plot the errors
    fig, ax = plt.subplots(nrows=1, ncols=3, num=None, figsize=(16, 6), dpi=90, facecolor='w', edgecolor='k')
    x = tuple(str(float(decimal.Decimal(sys.dt).normalize())) for sys in [sys_a, sys_b])
    fig.suptitle('Component errors for timesteps %s and %s' % x, fontsize=18, c='#282828')
    # # fig.text(.5, 0.8, 'Blah blah', ha='center')


    for i, key in enumerate(plot_keys):
        x = difference_dict[key].index.values
        y = difference_dict[key].values
        ax[i].plot(x, y)
        ax[i].set_facecolor('#f4f4f4')
        ax[i].set_xlabel('time (periods)')
        ax[i].set_title(key + ' error')
        ax[i].set_ylim(min_, max_)
        ax[i].set_xlim(0, max(difference_dict[key].index) + 0.1)
        ax[i].grid(which='major')
        # ax[i].set_yscale('symlog')
    ax[0].set_ylabel(nd_units['r'], fontsize=13.5)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, wspace=0.2, hspace=0.2)


def EXAMPLE_SINGLE_PLOT(system):
    fig = plt.figure(figsize=(12, 9))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    ax = fig.add_subplot(111, projection="3d")
    system.plot_system(ax, 'r_sol')
    plt.show()


def EXAMPLE_POS_VEL_PLOT(system):

    fig = plt.figure(figsize=(12, 9))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    ax = fig.add_subplot(121, projection="3d")
    system.plot_system(ax, 'r_sol')
    ax = fig.add_subplot(122, projection="3d")
    system.plot_system(ax, 'v_sol')

    plt.show()


def EXAMPLE_COMPARE_TIMESTEPS(systems):

        fig = plt.figure(figsize=(14, 8))
        fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
        fig.suptitle('Position of system at various time steps')

        ax1 = fig.add_subplot(121, projection="3d")
        systems[0].plot_system(ax1, 'r_sol')
        ax2 = fig.add_subplot(122, projection="3d")
        systems[1].plot_system(ax2, 'r_sol')

        fig2 = plt.figure(figsize=(14, 8))
        fig2.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
        fig2.suptitle('Position of system at various time steps')

        ax3 = fig2.add_subplot(121, projection="3d")
        systems[0].plot_system(ax3, 'v_sol')
        ax4 = fig2.add_subplot(122, projection="3d")
        systems[1].plot_system(ax4, 'v_sol')


def EXAMPLE_superimposed(systems):
    """
    Superimpose two solutions atop one another
    :param systems: a list of n-body-systems for which to superimpose
    :return:
    """
    fig = plt.figure(figsize=(12, 8))
    fig.set_tight_layout({"pad": 1, "w_pad": 1, 'h_pad': 1})
    fig.suptitle('Position of system at various time steps')

    ax1 = fig.add_subplot(111, projection="3d")
    ax1 = systems[0].plot_system(ax1, 'r_sol')
    ax1 = systems[1].plot_system(ax1, 'r_sol', linestyle='dashed')


def EXAMPLE_EXAMINE_ERRORS(systems):
    system_combinations = list(cmb(systems, 2))
    for cmb_ in system_combinations:
        plot_difference(*cmb_)



def main():

    # 1. Initialize Bodies
    alpha_centauri_a = NamedBody('Alpha Centauri A', m=1.1, r0=np.array([-0.5, 0, 0]), v0=np.array([0.01, 0.01, 0]))
    alpha_centauri_b = NamedBody('Alpha Centauri B', m=0.907, r0=np.array([0.5, 0, 0]), v0=np.array([-0.05, 0, -0.1]))
    body_c = NamedBody('Body C', m=1.0, r0=[0, 1, 0], v0=[0, -0.01, 0])
    body_d = NamedBody('Body D', m=0.8, r0=[0, 0, 1], v0=[0, 0.01, 0])
    two_bodies = [alpha_centauri_a, alpha_centauri_b]
    three_bodies = two_bodies + [body_c]

    # 2. Initialize each system
    input_bodies = three_bodies
    p, time_steps = 12,  [2**-int(v) for v in np.arange(3, 11, 2)]
    Systems = [NBodySystem(copy.deepcopy(input_bodies), nd, dt=step, periods=p)
               for step in time_steps]


    # 3. Solve each system w/ optional saving
    for i, system in enumerate(Systems):
        system.execute()
        system.save_solutions(save=False, dir_='results//',
                              tag='-' + str(len(system.bodies)) + '-' + str(p) + '-' + str(int(1/time_steps[i])))


    # fig, (ax1, ax2) = plt.subplots(ncols=2, subplot_kw={'projection':'3d'})

    # 4. PLOT EXAMPLES
    # 1. Plot a single set of solution data
    # EXAMPLE_SINGLE_PLOT(Systems[0])

    # 2. Animate one the solution
    # Systems[2].animate_solution('r_sol')

    # 3. Position velocity graph
    EXAMPLE_POS_VEL_PLOT(Systems[0])

    # 4. Several Figures: Compare position and velocity for two distinct time_step values
    # EXAMPLE_COMPARE_TIMESTEPS(Systems)

    # 5. Examine error btwn. solns. of diff. timesteps
    # EXAMPLE_EXAMINE_ERRORS(Systems[:2])

    # 6. Superimpose two systems to visualize the overlap
    # EXAMPLE_superimposed(Systems)
    plt.show()


if __name__ == "__main__":
    main()
    exit()


