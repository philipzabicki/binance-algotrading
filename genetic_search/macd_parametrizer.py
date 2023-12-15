from csv import writer
# from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from datetime import datetime as dt
from multiprocessing import Pool, cpu_count

from matplotlib import pyplot as plt
from numpy import array, min, mean, max, inf
# from pymoo.algorithms.moo.dnsga2 import DNSGA2
# from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, \
    MixedVariableDuplicateElimination
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.problem import StarmapParallelization
from pymoo.core.variable import Real, Integer
# from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP

from definitions import REPORT_DIR
from enviroments.macd_env import MACDStratSpotEnv
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

CPU_CORES_COUNT = cpu_count()
POP_SIZE = 1024
N_GEN = 10


class MACDMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, **kwargs):
        self.env = MACDStratSpotEnv(df=df,
                                    # max_steps=259_200,
                                    init_balance=300,
                                    no_action_finish=inf,
                                    fee=0.0,
                                    coin_step=0.00001,
                                    slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                                    verbose=False)
        macd_variables = {"stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['enter_at'], X['close_at'], X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
        _, reward, _, _, _ = self.env.step(action)
        # print(f'_evaluate() reward:{reward}')
        out["F"] = array([-reward])


class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.opt = []

    def notify(self, algorithm):
        # print(f'algorithm.pop.get("F") {algorithm.pop.get("F")}')
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew < 0:
            min_rew = min(algorithm.pop.get("F"))
            max_rew = max(algorithm.pop.get("F"))
            self.opt.append([min_rew, avg_rew, max_rew])
            # self.opt.append([avg_rew])
        else:
            self.opt.append([0.0, 0.0, 0.0])
        # print(self.opt)


def display_callback(callback, fname):
    plt.title("Convergence")
    plt.ylabel('Reward (min/avg/max)')
    plt.xlabel('Population')
    plt.plot(-1 * array(callback.opt), "--")
    plt.savefig(REPORT_DIR + 'Convergence_' + fname)
    return plt
    # plt.show()


def display_result(result, problem, fname):
    X_array = array([[entry['stop_loss'], entry['enter_at'], entry['close_at'],
                      entry['fast_period'], entry['slow_period'], entry['signal_period'],
                      entry['fast_ma_type'], entry['slow_ma_type'], entry['signal_ma_type']]
                     for entry in result.pop.get("X")])
    pop_size = len(X_array)
    labels = ['stop_loss', 'enter_at', 'close_at', 'fast_period', 'slow_period', 'signal_period', 'fast_ma_type',
              'slow_ma_type', 'signal_ma_type']
    bounds = array([problem.vars[name].bounds for name in labels]).T
    plot = PCP(labels=labels, bounds=bounds, n_ticks=10)
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
    plot.save(REPORT_DIR + fname)
    return plot
    # plot.show()


def main():
    _, df = by_BinanceVision(ticker='BTCFDUSD',
                             interval='1m',
                             market_type='spot',
                             data_type='klines',
                             start_date='2023-09-11',
                             split=True,
                             delay=0)
    # env = MACDStratSpotEnv(df=df,
    #                        # max_steps=259_200,
    #                        init_balance=300,
    #                        no_action_finish=inf,
    #                        fee=0.0,
    #                        coin_step=0.00001,
    #                        slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
    #                        verbose=False)

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)
    problem = MACDMixedVariableProblem(df, elementwise_runner=runner)

    # algorithm = NSGA2(pop_size=100)
    # algorithm = DNSGA2(pop_size=64)
    # algorithm = MixedVariableGA(pop=10)
    # algorithm = Optuna()
    algorithm = GA(pop_size=POP_SIZE,
                   sampling=MixedVariableSampling(),
                   mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                   eliminate_duplicates=MixedVariableDuplicateElimination())

    res = minimize(problem,
                   algorithm,
                   save_history=False,
                   callback=MyCallback(),
                   # termination=('n_gen', N_GEN),
                   termination=("time", "02:0:00"),
                   verbose=True)

    print(f'Exec time: {res.exec_time:.2f}s')
    # print('FINAL POPULATION')
    # print(f'res.pop.get("F") {res.pop.get("F")}')
    # print(f'res.pop.get("X") {res.pop.get("X")}')
    with open(REPORT_DIR + 'pop' + str(POP_SIZE) + str(dt.today()).replace(':', '-')[:-7] + '.csv', 'w',
              newline='') as file:
        csv_writer = writer(file)
        for f, x in zip(res.pop.get("F"), res.pop.get("X")):
            _row = [-1 * f[0], x['stop_loss'], x['enter_at'], x['close_at'], x['fast_period'], x['slow_period'],
                    x['signal_period'], x['fast_ma_type'], x['slow_ma_type'], x['signal_ma_type']]
            csv_writer.writerow(_row)
            print(f'writing row {_row}')
    # print(f'res.pop {res.pop}')
    # print(f'res.pop.get(X) {res.pop.get("X")}')
    # print(f'res.pop.get(F) {res.pop.get("F")}')
    if len(res.F) == 1:
        if isinstance(res.X, dict):
            print(
                f'Reward: {-res.f} Variables: {res.X["fast_period"], res.X["slow_period"], res.X["signal_period"], res.X["fast_ma_type"], res.X["slow_ma_type"], res.X["signal_ma_type"]}')
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
