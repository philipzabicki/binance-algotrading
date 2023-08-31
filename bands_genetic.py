import os
import numpy as np
import multiprocessing
#from multiprocessing.pool import ThreadPool
import get_data
from utility import minutes_since, get_slips_stats
from enviroments.BandsStratEnv import BandsStratEnvSpot
import matplotlib.pyplot as plt
from pymoo.core.problem import StarmapParallelization
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer
#from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pymoo.visualization.pcp import PCP
from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
#from pymoo.algorithms.soo.nonconvex.optuna import Optuna


CPU_CORES_COUNT = multiprocessing.cpu_count()-1
POP_SIZE = 16
#CPU_CORES_COUNT = 6

class CustomProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(n_var=5,
                         n_obj=1,
                         n_constr=0,
                         xl=np.array([0.0001, 0, 2, 1, 0.001]),
                         xu=np.array([0.0150, 32, 450, 500, 10.000]), **kwargs)
    def _evaluate(self, X, out, *args, **kwargs):
        _, reward, _, _ = self.env.step(X)
        out["F"] = np.array([-reward])

class CustomMixedVariableProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        vars = {"SL": Real(bounds=(0.0001, 0.0150)),
                "type": Integer(bounds=(0, 31)),
                "MAperiod": Integer(bounds=(2, 450)),
                "ATRperiod": Integer(bounds=(1, 500)),
                "ATRmulti": Real(bounds=(0.001, 10.000))}
        super().__init__(vars=vars, n_obj=1, **kwargs)
    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['SL'], X['type'], X['MAperiod'], X['ATRperiod'], X['ATRmulti']]
        #print(action)
        _, reward, _, info = self.env.step(action)
        out["F"] = np.array([-reward])

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt = []
    def notify(self, algorithm):
        #print(algorithm)
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt.append(algorithm.opt[0].F)


def display(result, problem, fname):
    n_evals = np.array([e.evaluator.n_eval for e in result.history])
    opt = np.array([-e.opt[0].F for e in result.history])
    plt.title("Convergence")
    plt.ylabel('Reward')
    plt.xlabel('n_evals')
    plt.plot(n_evals, opt, "--")
    plt.savefig(os.getcwd()+'/reports/Convergence_'+fname)

    #X_array = np.array([list(entry.values()) for entry in res.pop.get("X")])
    X_array = np.array([[entry['type'], entry['MAperiod'], entry['ATRperiod'], entry['ATRmulti'], entry['SL']] for entry in result.pop.get("X")])
    pop_size = len(X_array)
    #print(X_array)
    #labels = list(res.X.keys())
    labels = ['type','MAperiod','ATRperiod','ATRmulti','SL']
    bounds = np.array([problem.vars[name].bounds for name in labels]).T
    #X = np.array([[sol.X[name] for name in labels] for sol in res.opt])
    plot = PCP(labels=labels, bounds=bounds, n_ticks=10)
    plot.set_axis_style(color="grey", alpha=1)
    #plot.add(X_array[-1], color="black")
    #plot.add(X_array, color='#c5f8ff')
    plot.add(X_array[int(pop_size*.9)+1:], color='#a4f0ff')
    plot.add(X_array[int(pop_size*.8)+1:int(pop_size*.9)], color='#88e7fa')
    plot.add(X_array[int(pop_size*.7)+1:int(pop_size*.8)], color='#60d8f3')
    plot.add(X_array[int(pop_size*.6)+1:int(pop_size*.7)], color='#33c5e8')
    plot.add(X_array[int(pop_size*.5)+1:int(pop_size*.6)], color='#12b0da')
    plot.add(X_array[int(pop_size*.4)+1:int(pop_size*.5)], color='#019cc8')
    plot.add(X_array[int(pop_size*.3)+1:int(pop_size*.4)], color='#0086b4')
    plot.add(X_array[int(pop_size*.2)+1:int(pop_size*.3)], color='#00719f')
    plot.add(X_array[int(pop_size*.1)+1:int(pop_size*.2)], color='#005d89')
    plot.add(X_array[:int(pop_size*.1)], color='#004a73')
    plot.add(X_array[0], color='red')
    plot.save(os.getcwd()+'/reports/'+fname)
    plt.show()
    plot.show()

def main():
    df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False, statements=True, delay=3_600)
    df = df.drop(columns='Opened').to_numpy()
    df = np.hstack((df, np.zeros((df.shape[0], 1))))
    env = BandsStratEnvSpot(df=df[-minutes_since('23-03-2023'):,:].copy(), init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=get_slips_stats())

    pool = multiprocessing.Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)

    '''
    pool = ThreadPool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)
    '''

    problem = CustomMixedVariableProblem(env, elementwise_runner=runner)
    #algorithm = NSGA2(pop_size=100)
    #algorithm = DNSGA2(pop_size=64)
    #algorithm = MixedVariableGA(pop=10)
    algorithm = NSGA2(pop_size=POP_SIZE,
                      sampling=MixedVariableSampling(),
                      mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                      eliminate_duplicates=MixedVariableDuplicateElimination())
    #algorithm = Optuna() 

    res = minimize(problem,
                   algorithm,
                   save_history=True,
                   termination=('n_gen', 100),
                   verbose=True)

    print('Exec time:', res.exec_time)
    #print(f'res.pop.get(X) {res.pop.get("X")}')
    #print(f'res.pop.get(F) {res.pop.get("F")}')
    if len(res.F)==1:
        if isinstance(res.X, dict):
            print(f'Reward: {-res.f} Variables: {round(res.X["SL"],4), res.X["type"], res.X["MAperiod"], res.X["ATRperiod"], round(res.X["ATRmulti"],3)}')
            filename = f'Pop{POP_SIZE}Rew{-res.f:.0f}Vars{round(res.X["SL"],4):.4f}-{res.X["type"]}-{res.X["MAperiod"]}-{res.X["ATRperiod"]}-{round(res.X["ATRmulti"],3):.3f}.png'
        else:
            print(f'Reward: {-res.f} Variables: {round(res.X[0],4),int(res.X[1]),int(res.X[2]),int(res.X[3]),round(res.X[4],3)}')
            filename = 'Pop'+str(POP_SIZE)+'Rew'+str(-res.f)+'Vars'+str(round(res.X[0],4))+str(res.X[1])+str(res.X[2])+str(res.X[3])+str(round(res.X[4],3))+'.png'
    else:
        for front, var in zip(res.F, res.X):
            print(f"Reward:", front , "Variables:", round(var[0],4),int(var[1]),int(var[2]),int(var[3]),round(var[4],3))
            filename = 'Figure.png'
    
    display(res, problem, filename)

if __name__ == '__main__':
    main()
