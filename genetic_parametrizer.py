from datetime import datetime as dt
from multiprocessing import Pool, cpu_count

from numpy import inf
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, \
    MixedVariableDuplicateElimination
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize

from genetic_search.base import save_results, get_callback_plot, get_variables_plot, \
    MinAvgMaxNonzeroSingleObjCallback
from genetic_search.macd_parametrizer import MACDSavingFuturesMixedVariableProblem
from genetic_search.stoch_parametrizer import StochSavingFuturesMixedVariableProblem
from genetic_search.chaikinosc_parametrizer import ChaikinOscillatorSavingFuturesMixedVariableProblem
from genetic_search.bands_parametrizer import BandsSavingFuturesMixedVariableProblem
from utils.get_data import by_BinanceVision

CPU_CORES_COUNT = cpu_count()
# CPU_CORES_COUNT = 1
POP_SIZE = 1024
N_GEN = 175
TICKER = 'BTCUSDT'
ITV = '5m'
MARKET_TYPE = 'um'
DATA_TYPE = 'klines'
START_DATE = '2021-08-30'
END_DATE = '2021-10-18'
PROBLEM = MACDSavingFuturesMixedVariableProblem
PROBLEM_N_EVALS = 1
PROBLEM_METRIC = 'mean'
ALGORITHM = NSGA2
#TERMINATION = ("time", "02:00:00")
TERMINATION = ('n_gen', N_GEN)
ENV_KWARGS = {#'max_steps': 2_016,
              'init_balance': 50,
              'no_action_finish': inf,
              'fee': 0.0005,
              'coin_step': 0.001,
              # 'slippage': get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              'verbose': False}


def main():
    # _, df = by_BinanceVision(ticker='BTCFDUSD',
    #                          interval='1m',
    #                          market_type='spot',
    #                          data_type='klines',
    #                          start_date='2023-09-11',
    #                          split=True,
    #                          delay=0)
    # print(f'df used: {df}')
    _, df = by_BinanceVision(ticker=TICKER,
                             interval=ITV,
                             market_type=MARKET_TYPE,
                             data_type=DATA_TYPE,
                             start_date=START_DATE,
                             end_date=END_DATE,
                             split=True,
                             delay=345_600)
    print(f'df used: {df}')
    _, df_mark = by_BinanceVision(ticker=TICKER,
                                  interval=ITV,
                                  market_type=MARKET_TYPE,
                                  data_type='markPriceKlines',
                                  start_date=START_DATE,
                                  end_date=END_DATE,
                                  split=True,
                                  delay=345_600)
    print(f'df_mark used: {df_mark}')

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)
    # MACDStratSpotEnv init arguments:
    # env_kwargs = {'init_balance': 350,
    #               'no_action_finish': inf,
    #               'fee': 0.0,
    #               'coin_step': 0.00001,
    #               'slippage': get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
    #               'verbose': False}
    problem = PROBLEM(df,
                      df_mark,
                      env_kwargs=ENV_KWARGS,
                      n_evals=PROBLEM_N_EVALS,
                      metric=PROBLEM_METRIC,
                      elementwise_runner=runner)

    algorithm = ALGORITHM(pop_size=POP_SIZE,
                          sampling=MixedVariableSampling(),
                          mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                          eliminate_duplicates=MixedVariableDuplicateElimination())

    _date = str(dt.today()).replace(":", "-")[:-7]
    dir_name = f'{TICKER}{ITV}_{MARKET_TYPE}_Pop{POP_SIZE}_{problem.env.__class__.__name__}_{_date}'
    res = minimize(problem,
                   algorithm,
                   save_history=False,
                   # callback=GenerationSavingCallback(problem, dir_name, verbose=True),
                   callback=MinAvgMaxNonzeroSingleObjCallback(problem, verbose=True),
                   termination=TERMINATION,
                   verbose=True)

    print(f'Exec time: {res.exec_time:.2f}s')
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
            print(f"front=", front, "variables=", list(var.values()))

    plt1 = get_callback_plot(res.algorithm.callback, filename)
    plt2 = get_variables_plot(res.pop.get("X"), problem, filename)
    # Shows both:
    plt1.show()


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # profiler.print_stats(sort='tottime')
