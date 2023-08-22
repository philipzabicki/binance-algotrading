import numpy as np
from datetime import datetime
from dateutil.parser import parse
import multiprocessing
#from multiprocessing.pool import ThreadPool
from pymoo.core.problem import StarmapParallelization

from enviroments.BandParametrizerEnv import BandParametrizerEnv
import get_data

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.algorithms.moo.dnsga2 import DNSGA2


def minutes_since(data_string):
    diff = datetime.now() - parse(data_string)
    minutes = diff.total_seconds() / 60
    return int(minutes)

class CustomProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(n_var=5,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([0.0001, 0, 1, 1, 0.001]),
                         xu=np.array([0.015, 33, 200, 500, 5]), **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        _, reward, _, _ = self.env.step(X)
        out["F"] = np.array([-reward])

def main():
    SLIPPAGES = {'market_buy':(1.000022, 0.000045), 'market_sell':(0.999971, 0.000035), 'SL':(1.0, 0.000002)}
    df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=300)
    df = df.drop(columns='Opened').to_numpy()
    df = np.hstack((df, np.zeros((df.shape[0], 1))))
    env = BandParametrizerEnv(df=df[-minutes_since('23-03-2023'):,:].copy(), init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPPAGES)

    n_proccess = multiprocessing.cpu_count()//2
    pool = multiprocessing.Pool(n_proccess)
    runner = StarmapParallelization(pool.starmap)

    '''n_proccess = multiprocessing.cpu_count()//2
    pool = ThreadPool(n_threads)
    runner = StarmapParallelization(pool.starmap)'''

    problem = CustomProblem(env, elementwise_runner=runner)

    #algorithm = NSGA2(pop_size=100)

    algorithm = DNSGA2(pop_size=256)

    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', 50),
                   seed=1,
                   verbose=True)

    print('Exec time:', res.exec_time)
    if len(res.F)==1:
        print(f'Reward {-res.f} Variables {round(res.X[0],4),int(res.X[1]),int(res.X[2]),int(res.X[3]),round(res.X[4],3)}')
    else:
        for front, var in zip(res.F, res.X):
            print(f"Reward", front , "Variables:", round(var[0],4),int(var[1]),int(var[2]),int(var[3]),round(var[4],3))


if __name__ == '__main__':
    main()

'''res = minimize(problem,
               algorithm,
               ("n_gen", 100),
               verbose=True,
               save_history=True)'''