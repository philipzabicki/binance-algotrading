from datetime import datetime as dt
from multiprocessing import Pool, cpu_count
import os

from numpy import inf, mean
import numpy as np
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.sampling import Sampling
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize

from genetic_search.base import save_results
from genetic_search.action_parametrizer import SpotActionVariablesProblem
from enviroments.signal_env import SignalExecuteSpotEnv
from utils.get_data import by_BinanceVision


class CustomIntegerSampling(Sampling):
    def __init__(self, probabilities=None):
        super().__init__()
        # Domyślne prawdopodobieństwa: 0 ma 50%, 1 i 2 mają po 25%
        if probabilities is None:
            self.probabilities = [0.4, 0.3, 0.3]
        else:
            self.probabilities = probabilities
        self.values = np.arange(len(self.probabilities))

    def _do(self, problem, n_samples, **kwargs):
        # problem.n_var to liczba genów w chromosomie
        samples = np.random.choice(
            self.values,
            size=(n_samples, problem.n_var),
            p=self.probabilities
        )
        return samples


CPU_CORES_COUNT = cpu_count()

TICKER = 'BTCUSDT'
ITV = '1h'
MARKET_TYPE = 'spot'
DATA_TYPE = 'klines'
PROBLEM = SpotActionVariablesProblem
ALGORITHM = GA
POP_SIZE = 512
# TERMINATION = ("time", "01:00:00")
TERMINATION = ('n_gen', 300)

ENV_KWARGS = {
    'max_steps': 0,
    'init_balance': 10_000,
    'no_action_finish': inf,
    'fee': 0.00075,
    'coin_step': 0.00001,
    'verbose': False
}

N_PARTS = 100
results_dir = "results_parts"

def save_actions_with_dates(df, res, part_number=None, output_dir="results_parts"):
    # Zakładamy, że res.X to np.array z najlepszym rozwiązaniem i ma tyle samo elementów co df
    actions = res.X  # np.array([-1,-1,1,0,0])
    df['action'] = actions

    os.makedirs(output_dir, exist_ok=True)

    if part_number is not None:
        filename = f"part_{part_number}_actions_rew{-res.F[0]}.csv"
    else:
        filename = "actions.csv"

    filepath = os.path.join(output_dir, filename)
    df[['Opened', 'action']].to_csv(filepath, index=False)
    print(f"Wyniki zapisane do {filepath}")


def main():
    print(f'CPUs used: {CPU_CORES_COUNT}')
    df_full = by_BinanceVision(ticker=TICKER,
                               interval=ITV,
                               market_type=MARKET_TYPE,
                               data_type=DATA_TYPE,
                               split=False,
                               delay=100_000)
    print(f'Full df length: {len(df_full)}')

    length = len(df_full)
    part_size = length // N_PARTS

    os.makedirs(results_dir, exist_ok=True)

    all_rewards = []

    for i in range(72, N_PARTS):
        start_idx = i * part_size
        end_idx = (i + 1) * part_size if i < N_PARTS - 1 else length

        df = df_full.iloc[start_idx:end_idx].copy()
        print(f'Processing part {i+1}/{N_PARTS}, df shape: {df.shape}')
        print(f'start_idx {start_idx} end_idx: {end_idx}')

        pool = Pool(CPU_CORES_COUNT)
        runner = StarmapParallelization(pool.starmap)
        problem = PROBLEM(df,
                          env_kwargs=ENV_KWARGS,
                          elementwise_runner=runner)

        algorithm = ALGORITHM(pop_size=POP_SIZE,
                              sampling=IntegerRandomSampling(),
                              crossover=UniformCrossover(repair=RoundingRepair()),
                              mutation=PolynomialMutation(eta=1, repair=RoundingRepair()),
                              eliminate_duplicates=True)

        _date = str(dt.today()).replace(":", "-")[:-7]

        res = minimize(problem,
                       algorithm,
                       save_history=False,
                       termination=TERMINATION,
                       verbose=True)

        pool.close()
        pool.join()

        print(f'Exec time for part {i+1}: {res.exec_time:.2f}s')
        print(f'res.F {res.F} res.X {res.X}')

        filename = f'{TICKER}{ITV}_{MARKET_TYPE}_Pop{POP_SIZE}_part_{i+1}_ngen{res.algorithm.n_iter - 1}_{problem.env.__class__.__name__}_{_date}'
        filepath = os.path.join(results_dir, filename)

        save_results(filepath, res)

        # Wyświetlanie informacji o najlepszym rozwiązaniu:
        if len(res.F) == 1:
            best_f = res.F[0]
            best_reward = -best_f
            print(f'Best gene for part {i+1}: reward= {best_reward} variables= {res.X}')
        else:
            print('Pareto front for part {i+1}:')
            for front, var in zip(res.F, res.X):
                print(f"front={front}, variables={var}")

        all_rewards.extend(res.F.flatten())

        # Zapisujemy dane w formacie Opened, action dla najlepszych parametrów:
        save_actions_with_dates(df, res, part_number=i+1, output_dir=results_dir)

    print(f'All parts done. Avg reward: {mean(all_rewards)}')

    ENV_KWARGS['verbose'] = True
    ENV_KWARGS['visualize'] = True
    env = SignalExecuteSpotEnv(df=df_full, **ENV_KWARGS)
    env.reset()
    env.signals = res.X
    env()


if __name__ == '__main__':
    main()
