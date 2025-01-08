import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from pymoo.core.problem import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.optuna import Optuna
from pymoo.core.mixed import MixedVariableGA
import matplotlib.pyplot as plt
from pymoo.core.mixed import MixedVariableMating, MixedVariableSampling, \
    MixedVariableDuplicateElimination

from genetic_search.feature_action_fitter import (
    MACDVarSourceFitting,
    KeltnerChannelFitting,
    StochSpotMixedVariableProblem,
    ChaikinOscillatorSpotMixedVariableProblem,
    KeltnerChannelVarSourceFitting
)

import os

CPU_CORES_COUNT = cpu_count() - 1
print(f'CPUs used: {CPU_CORES_COUNT}')
PROBLEM = MACDVarSourceFitting
ALGORITHM = MixedVariableGA
POP_SIZE = 4096
TERMINATION = ('n_gen', 50)

def convert_variables(variables):
    """
    Konwertuje listę krotek na słownik.
    """
    if isinstance(variables, dict):
        return variables
    elif isinstance(variables, list):
        return dict(variables)
    else:
        raise ValueError("Unsupported format for variables")

def main():
    df = pd.read_csv('results_parts/final_combined_actions.csv')
    print(f'df used:  {df}')
    # vals = df.loc[df['Weight'] > 1.0]
    # print(f'vals {vals}')
    quantiles = list(pd.qcut(df.loc[df['Weight'] > 1.0]['Weight'], q=10, labels=False, retbins=True)[1])
    print("Granice binów:")
    print(quantiles)

    # bins = pd.cut(df.loc[df['Weight'] > 1.0]['Weight'], q=10, include_lowest=True, right=True)
    # bin_counts = bins.value_counts().sort_index()
    # print(f'bin_counts {bin_counts}')

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)

    output_dir = r'C:\Cloud\filips19mail@gmail.com\github\binance-algotrading\reports\macd_action_fits'
    os.makedirs(output_dir, exist_ok=True)

    results = []  # Lista do przechowywania wyników dla danego ma_type
    for lower, upper in zip(quantiles[:-1], quantiles[1:]):
        print(f'Optimize run for macd, range: ({lower}, {upper})')
        problem = PROBLEM(
            df,
            lower,
            upper,
            elementwise_runner=runner
        )

        algorithm = ALGORITHM(
            n_jobs=-1,
            pop_size=POP_SIZE,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

        res = minimize(
            problem,
            algorithm,
            save_history=False,
            # callback=GenerationSavingCallback(problem, dir_name, verbose=True),
            # callback=MinAvgMaxNonzeroSingleObjCallback(problem, verbose=True),
            termination=TERMINATION,
            verbose=True
        )

        if len(res.F) == 1:
            variables_dict = convert_variables(res.X)
            best_gene = {
                'lower': lower,
                'upper': upper,
                'reward': -res.F[0]
            }
            best_gene.update(variables_dict)
            print(f'Best gene: {best_gene}')
            results.append(best_gene)
        else:
            for front, var in zip(res.F, res.X):
                variables_dict = convert_variables(var)
                pareto_result = {
                    'lower': lower,
                    'upper': upper,
                    'reward': -front
                }
                pareto_result.update(variables_dict)
                print(f"Pareto front: {pareto_result}")
                results.append(pareto_result)
        output_file = os.path.join(output_dir, f'results_macd.csv')
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f'Results for ma_type saved to {output_file}')
    pool.close()
    pool.join()

if __name__ == '__main__':
    main()
