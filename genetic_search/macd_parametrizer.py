from numpy import array, hstack, mean, zeros, arange, inf
from multiprocessing import Pool, cpu_count
from get_data import by_DataClient, by_BinanceVision
from matplotlib import pyplot as plt
from pymoo.core.problem import StarmapParallelization
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
# from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
# from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, \
    MixedVariableDuplicateElimination
# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from datetime import datetime as dt
from csv import writer
from enviroments.macd import MACDStratEnv
from definitions import REPORT_DIR
from utilityx import minutes_since, get_market_slips_stats

# from time import time
# import cProfile
# from gc import collect


CPU_CORES_COUNT = cpu_count()
POP_SIZE = 4096
N_GEN = 5


# print(SLIPP)
# CPU_CORES_COUNT = 6

class MACDProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(n_var=6,
                         n_obj=1,
                         n_constr=0,
                         xl=array([2, 2, 2, 0, 0, 0]),
                         xu=array([10_000, 10_000, 10_000, 9, 9, 9]),
                         **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        _, reward, _, _ = self.env.step(X)
        out["F"] = array([-reward])

class MACDMixedVariableProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        macd_variables = {"fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 8)),
                          "slow_ma_type": Integer(bounds=(0, 8)),
                          "signal_ma_type": Integer(bounds=(0, 8))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)
    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['fast_period'], X['slow_period'], X['signal_period'], X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
        _, reward, _, _, _ = self.env.step(action)
        out["F"] = array([-reward])


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        # self.opt = zeros(N_GEN)
        # self.idx = 0
        self.opt = []

    def notify(self, algorithm):
        avg_rew = mean(algorithm.pop.get("F"))
        # print(f'Current population: {algorithm.pop.get("F")}')
        if (avg_rew != inf) and (avg_rew != -inf) and (avg_rew > 0):
            self.opt.append(avg_rew)
        # print(f'avg_rew {avg_rew}')
        # print(f'self.opt {self.opt}')
        # self.opt[self.idx] = avg_rew if avg_rew<0 else 0.0
        # self.opt[self.idx] = avg_rew
        # self.idx += 1


def display_callback(callback, fname):
    plt.title("Convergence")
    plt.ylabel('Mean Reward')
    plt.xlabel('Population')
    plt.plot(-1 * array(callback.opt), "--")
    plt.savefig(REPORT_DIR + 'Convergence_' + fname)
    return plt
    # plt.show()


def display_result(result, problem, fname):
    X_array = array([[entry['fast_period'], entry['slow_period'], entry['signal_period'],
                      entry['fast_ma_type'], entry['slow_ma_type'], entry['signal_ma_type']]
                     for entry in result.pop.get("X")])
    print(X_array)
    pop_size = len(X_array)
    labels = ['fast_period', 'slow_period', 'signal_period', 'fast_ma_type', 'slow_ma_type', 'signal_ma_type']
    bounds = array([problem.vars[name].bounds for name in labels]).T
    plot = PCP(labels=labels, n_ticks=10)
    plot.set_axis_style(color="grey", alpha=1)
    plot.add(X_array, color="grey", alpha=0.3)
    '''plot.add(X_array[int(pop_size * .9) + 1:], linewidth=1.9, color='#a4f0ff')
    plot.add(X_array[int(pop_size * .8) + 1:int(pop_size * .9)], linewidth=1.8, color='#88e7fa')
    plot.add(X_array[int(pop_size * .7) + 1:int(pop_size * .8)], linewidth=1.7, color='#60d8f3')
    plot.add(X_array[int(pop_size * .6) + 1:int(pop_size * .7)], linewidth=1.6, color='#33c5e8')
    plot.add(X_array[int(pop_size * .5) + 1:int(pop_size * .6)], linewidth=1.5, color='#12b0da')
    plot.add(X_array[int(pop_size * .4) + 1:int(pop_size * .5)], linewidth=1.4, color='#019cc8')
    plot.add(X_array[int(pop_size * .3) + 1:int(pop_size * .4)], linewidth=1.3, color='#0086b4')
    plot.add(X_array[int(pop_size * .2) + 1:int(pop_size * .3)], linewidth=1.2, color='#00719f')
    plot.add(X_array[int(pop_size * .1) + 1:int(pop_size * .2)], linewidth=1.1, color='#005d89')
    plot.add(X_array[:int(pop_size * .1)], linewidth=1.0, color='#004a73')
    plot.add(X_array[0], linewidth=1.5, color='red')'''
    plot.save(REPORT_DIR + fname)
    return plot
    # plot.show()


def main():
    # pool = ThreadPool(CPU_CORES_COUNT)
    # runner = StarmapParallelization(pool.starmap)
    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)

    # df = by_DataClient(ticker='BTCFDUSD', interval='1s', futures=False, statements=True, delay=3_600)
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1m', type='spot', data='klines', delay=129_600)
    df = df.drop(columns='Opened').to_numpy()[-minutes_since('11-09-2023'):, :]
    df = hstack((df, zeros((df.shape[0], 1))))
    # df = df[-seconds_since('09-01-2023'):, :]
    print(df)
    env = MACDStratEnv(df=df, init_balance=1_000, no_action_finish=inf, fee=0.0, coin_step=0.00001, slippage=get_market_slips_stats(), verbose=False)
    problem = MACDMixedVariableProblem(env, elementwise_runner=runner)
    # algorithm = NSGA2(pop_size=100)
    # algorithm = DNSGA2(pop_size=64)
    # algorithm = MixedVariableGA(pop=10)
    algorithm = GA(pop_size=POP_SIZE,
                   sampling=MixedVariableSampling(),
                   mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                   eliminate_duplicates=MixedVariableDuplicateElimination())
    # algorithm = Optuna()

    res = minimize(problem,
                   algorithm,
                   save_history=False,
                   callback=MyCallback(),
                   termination=('n_gen', N_GEN),
                   # termination=("time", "09:00:00"),
                   verbose=True)

    print(f'Exec time: {res.exec_time:.2f}s')
    print('FINAL POPULATION')
    print(f'res.pop.get("F") {res.pop.get("F")}')
    print(f'res.pop.get("X") {res.pop.get("X")}')
    with open(REPORT_DIR + 'pop' + str(POP_SIZE) + str(dt.today()).replace(':', '-')[:-7] + '.csv', 'w',
              newline='') as file:
        csv_writer = writer(file)
        for f, x in zip(res.pop.get("F"), res.pop.get("X")):
            _row = [*f, x['fast_period'], x['slow_period'], x['signal_period'], x['fast_ma_type'], x['slow_ma_type'], x['signal_ma_type']]
            csv_writer.writerow(_row)
            print(f'writing row {_row}')
    # print(f'res.pop {res.pop}')
    # print(f'res.pop.get(X) {res.pop.get("X")}')
    # print(f'res.pop.get(F) {res.pop.get("F")}')
    if len(res.F) == 1:
        if isinstance(res.X, dict):
            print(f'Reward: {-res.f} Variables: {res.X["fast_period"], res.X["slow_period"], res.X["signal_period"], res.X["fast_ma_type"], res.X["slow_ma_type"], res.X["signal_ma_type"]}')
            filename = f'Pop{POP_SIZE}Rew{-res.f:.0f}Vars{res.X["fast_period"]}-{res.X["slow_period"]}-{res.X["signal_period"]}-{res.X["fast_ma_type"]}-{res.X["slow_ma_type"]}-{res.X["signal_ma_type"]}.png'
        else:
            print(
                f'Reward: {-res.f} Variables: {round(res.X[0], 4), int(res.X[1]), int(res.X[2]), int(res.X[3]), round(res.X[4], 3)}')
            filename = 'Pop' + str(POP_SIZE) + 'Rew' + str(-res.f) + 'Vars' + str(round(res.X[0], 4)) + str(
                res.X[1]) + str(res.X[2]) + str(res.X[3]) + str(round(res.X[4], 3)) + '.png'
    else:
        for front, var in zip(res.F, res.X):
            print(f"Reward:", front, "Variables:", var)
        filename = 'Figure.png'

    plt1 = display_callback(res.algorithm.callback, filename)
    plt2 = display_result(res, problem, filename)
    # plt.subplot(plt1)
    # plt1.show()
    plt1.show()
    # plt1.clf()
    # plt2.show()
    # time(1)
    # plt2.show()


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()  # Zakończ profilowanie
    # profiler.print_stats(sort='tottime')