import os
from csv import writer

from matplotlib import pyplot as plt
from numpy import array, min, mean, max
from pymoo.core.callback import Callback
from pymoo.visualization.pcp import PCP

from definitions import REPORT_DIR


def save_results(filename, result):
    filename = REPORT_DIR + filename + '.csv'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', newline='') as file:
        _header = ['reward'] + [*result.pop.get("X")[0].keys()]
        writer(file).writerow(_header)
        for f, x in zip(result.pop.get("F"), result.pop.get("X")):
            _row = [-1 * f[0], *x.values()]
            writer(file).writerow(_row)
            print(f'writing row {_row}')


def get_callback_plot(callback, fname):
    plt.figure(figsize=(16, 9))
    plt.title("Convergence")
    plt.ylabel('Reward (min/avg/max)')
    plt.xlabel('Population')
    plt.plot(-1 * array(callback.opt), "--")
    plt.savefig(REPORT_DIR + 'Convergence_' + fname + ".png", dpi=300)
    return plt


def get_variables_plot(x_variables, problem, fname, save=True):
    sample_gen = x_variables[0]
    if not isinstance(sample_gen, dict):
        raise NotImplementedError("Currently supports only list of dict type populations")
    X_array = array([[*entry.values()] for entry in x_variables])
    pop_size = len(X_array)
    labels = [*sample_gen.keys()]
    bounds = array([problem.vars[name].bounds for name in labels]).T
    plot = PCP(figsize=(16, 9), labels=labels, bounds=bounds, tight_layout=True)
    plot.set_axis_style(color="grey", alpha=1)
    plot.add(X_array, color="grey", alpha=0.3)
    plot.add(X_array[int(pop_size * .9) + 1:], linewidth=1.9, color='#a4f0ff')
    plot.add(X_array[int(pop_size * .8) + 1:int(pop_size * .9)], linewidth=1.8, color='#88e7fa')
    plot.add(X_array[int(pop_size * .7) + 1:int(pop_size * .8)], linewidth=1.7, color='#60d8f3')
    plot.add(X_array[int(pop_size * .6) + 1:int(pop_size * .7)], linewidth=1.6, color='#33c5e8')
    plot.add(X_array[int(pop_size * .5) + 1:int(pop_size * .6)], linewidth=1.5, color='#12b0da')
    plot.add(X_array[int(pop_size * .4) + 1:int(pop_size * .5)], linewidth=1.4, color='#019cc8')
    plot.add(X_array[int(pop_size * .3) + 1:int(pop_size * .4)], linewidth=1.3, color='#0086b4')
    plot.add(X_array[int(pop_size * .2) + 1:int(pop_size * .3)], linewidth=1.2, color='#00719f')
    plot.add(X_array[int(pop_size * .1) + 1:int(pop_size * .2)], linewidth=1.1, color='#005d89')
    plot.add(X_array[:int(pop_size * .1)], linewidth=1.0, color='#004a73')
    plot.add(X_array[0], linewidth=1.5, color='red')
    if save:
        plot.save(REPORT_DIR + fname + ".png", dpi=300)
    return plot


class MinAvgMaxNonzeroSingleObjCallback(Callback):
    def __init__(self, problem) -> None:
        super().__init__()
        self.opt = []
        self.problem = problem

    def notify(self, algorithm):
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            min_rew = min(algorithm.pop.get("F"))
            max_rew = max(algorithm.pop.get("F"))
            self.opt.append([min_rew, avg_rew, max_rew])
        else:
            self.opt.append([0.0, 0.0, 0.0])


class AverageNonzeroSingleObjCallback(Callback):
    def __init__(self, problem) -> None:
        super().__init__()
        self.opt = []
        self.problem = problem

    def notify(self, algorithm):
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            self.opt.append(avg_rew)
        else:
            self.opt.append(0.0)
