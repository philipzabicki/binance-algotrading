from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
import pandas as pd

from numpy import inf
# from pymoo.algorithms.moo.dnsga2 import DNSGA2
# from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, \
    MixedVariableDuplicateElimination
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize

from genetic_search.base import SingleObjNonzeroMinAvgMaxCallback, save_results, get_callback_plot, get_variables_plot
from genetic_search.chosc_parametrizer import ChaikinOscillatorMixedVariableProblem
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

CPU_CORES_COUNT = cpu_count()
# CPU_CORES_COUNT = 1
POP_SIZE = 2048
N_GEN = 100
TICKER = 'LOTOS'
ITV = '1d'
MARKET_TYPE = 'spot'
DATA_TYPE = 'klines'
START_DATE = '2023-09-11'


def main():
    # _, df = by_BinanceVision(ticker='BTCFDUSD',
    #                          interval='1m',
    #                          market_type='spot',
    #                          data_type='klines',
    #                          start_date='2023-09-11',
    #                          split=True,
    #                          delay=0)
    # print(f'df used: {df}')
    df = pd.read_csv("C:/github/binance-algotrading/.other/lotos.csv")
    df.drop(columns='Opened', inplace=True)
    print(f'df used: {df}')

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)
    # MACDStratSpotEnv init arguments:
    # env_kwargs = {'init_balance': 350,
    #               'no_action_finish': inf,
    #               'fee': 0.0,
    #               'coin_step': 0.00001,
    #               'slippage': get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
    #               'verbose': False}
    env_kwargs = {'init_balance': 1_000,
                  'no_action_finish': inf,
                  'fee': 0.0,
                  'coin_step': 1.0,
                  'verbose': False}
    problem = ChaikinOscillatorMixedVariableProblem(df,
                                                    env_kwargs=env_kwargs,
                                                    elementwise_runner=runner)

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
                   callback=SingleObjNonzeroMinAvgMaxCallback(problem),
                   termination=('n_gen', N_GEN),
                   # termination=("time", "00:05:00"),
                   verbose=True)

    print(f'Exec time: {res.exec_time:.2f}s')
    _date = str(dt.today()).replace(":", "-")[:-7]
    filename = f'{TICKER}{ITV}_{MARKET_TYPE}_Pop{POP_SIZE}_ngen{res.algorithm.n_iter - 1}_{problem.env.__class__.__name__}_{_date}'
    save_results(filename, res)

    if len(res.F) == 1:
        if isinstance(res.X, dict):
            print(f'Best gene: reward= {-res.f} variables= {list(res.X.values())}')
        else:
            print(f'Best gene: reward= {-res.f} variables= {res.X}')
    else:
        print('Pareto front:')
        for front, var in zip(res.F, res.X):
            print(f"front=", front, "variables=", var)

    plt1 = get_callback_plot(res.algorithm.callback, filename)
    plt2 = get_variables_plot(res.pop.get("X"), problem, filename)
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
