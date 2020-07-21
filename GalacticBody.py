import numpy as np
import os
import pandas as pd
import io
import copy

def saveTupleList(tuple_list, path):
    with io.open(path, 'w', encoding='utf-8') as f:
        f.writelines([','.join([str(item) for item in tuple]) + '\n' for tuple in tuple_list])


class GalacticBody:

    def __init__(self, m, r0, v0):

        self.m = m
        self.r0 = r0        # initial position
        self.v0 = v0        # initial velocity

        self.r = []         # current position
        self.v = []         # current velocity

        self.data_ = {
                'r': {'x': [], 'y': [], 'z': []},
                'v': {'dx': [], 'dy': [], 'dz': []},
                'r_com': {'x': [], 'y': [], 'z': []},
                'v_com': {'dx': [], 'dy': [], 'dz': []}
        }

        # Absolute position / velocity
        self.r_sol = {'x': [], 'y': [], 'z': []}
        self.v_sol = {'dx': [], 'dy': [], 'dz': []}

        # pos & vel rel. to system CoM
        self.r_com = {'x': [], 'y': [], 'z': []}
        self.v_com = {'dx': [], 'dy': [], 'dz': []}

        self.t = []

    def __repr__(self):
        return '\t'.join([str(self.m), str(self.r0), str(self.v0)])

    def plot(self, ax, key='r_sol', *args, **kwargs):
        """ Plot a solution"""
        plot_vals = self.__getattribute__(key)

        # Plot path
        ax.plot(*plot_vals.values(), lw=2.1, label=self.name, **kwargs)
        # Current Val
        ax.scatter(*[s[-1] for s in plot_vals.values()], marker="o", s=100, label=self.name, **kwargs)
        # Initial Val
        ax.scatter(*[s[0] for s in plot_vals.values()],  marker="x", s=100, **kwargs)
        return ax

    def reset(self):
        self.r_sol = {'x': [], 'y': [], 'z': []}
        self.v_sol = {'dx': [], 'dy': [], 'dz': []}


    def update_solutions(self, r_solns=None, v_solns=None, t=None):
        """

        :param r_solns:
        :param v_solns:
        :param t:
        :return:
        """


        self.t.extend(t)
        if r_solns is not None:
            for i, key in enumerate(self.r_sol.keys()):
                self.r_sol[key].extend(r_solns[i])



        if v_solns is not None:
            for i, key in enumerate(self.v_sol.keys()):
                self.v_sol[key].extend(v_solns[i])

    def to_df(self):
        """
        Convert the body data into a data-frame indexed by time
        :return: pandas dataframe

        """
        df = pd.DataFrame(dict(list(self.r_sol.items()) + list(self.v_sol.items())))
        df['t'] = self.t
        df.set_index('t', inplace=True)
        return df


class NamedBody(GalacticBody):

    def __init__(self, name, *args, **kwargs):
        super(NamedBody, self).__init__(*args, **kwargs)
        self.name = name

    def __repr__(self):
        return '\t'.join((self.name, super(NamedBody, self).__repr__()))

    def save_solution(self, tag=None, dir_=''):
        """
        :param tag: name of experiment (ex: tag = '-50-1000' if  50 periods 1000 pts per period)
        :param dir_: path to save firectory
        :return:

        """
        if tag == None:
            tag = ''

        d = {}
        d.update(self.r_sol)
        d['t'] = self.t
        d['norm'] = [np.linalg.norm(item) for item in list(zip(*(self.r_sol.values())))]
        p = os.path.join(dir_, self.name + '-r_sol' + tag + '.csv')
        saveTupleList([tuple(d.keys())] + list(zip(*d.values())), p)

        d.update(self.v_sol)
        d['t'] = self.t
        d['norm'] = [np.linalg.norm(item) for item in list(zip(*(self.v_sol.values())))]
        p = os.path.join(dir_, self.name + '-v_sol' + tag + '.csv')
        saveTupleList([tuple(d.keys())] + list(zip(*d.values())), p)

    def save_solution_df(self, dir_=None, tag=None):
        """
        Use pandas dataframes to compute norm and save to csv
        *note: this is much slower than the method save_solution, mostly because
        df.apply is much slower than working directly on the values
        :param tag:
        :return:
        """
        if tag == None:
            tag = ''
        df = self.to_df()
        df.to_csv('results//' + self.name + '-solution-' + tag + '.csv')


        # df['t'] = self.t

        # df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        # df = pd.DataFrame(self.v_sol)
        # df['t'] = self.t
        # df['norm'] = df.apply(lambda row: np.linalg.norm((row['x'], row['y'], row['z'])), axis=1)
        # df.to_csv('results//' + self.name + '-v_sol'  + tag +  '.csv')