from os import getcwd
from numpy import array, hstack, mean, zeros, arange, inf
from multiprocessing import Pool, cpu_count
from get_data import by_DataClient, by_BinanceVision
from utility import minutes_since, seconds_since, get_limit_slips_stats, get_market_slips_stats
from enviroments.bands import BandsStratEnv
from matplotlib import pyplot as plt
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
from datetime import datetime as dt
#from time import time
from os import getcwd
#import cProfile
#from gc import collect


CPU_CORES_COUNT = cpu_count()
POP_SIZE = 32
N_GEN = 500
SLIPP = get_market_slips_stats()
#print(SLIPP)
#CPU_CORES_COUNT = 6

class CustomProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        self.env = env
        super().__init__(n_var=7,
                         n_obj=1,
                         n_constr=0,
                         xl=array([0.0001, 0.001, 0.001, 0, 2, 1, 0.001]),
                         xu=array([0.0150, 1.000, 1.000, 35, 450, 500, 9.000]),
                         **kwargs)
    def _evaluate(self, X, out, *args, **kwargs):
        _, reward, _, _ = self.env.step(X)
        out["F"] = array([-reward])

class CustomMixedVariableProblem(ElementwiseProblem):
    def __init__(self, env, **kwargs):
        #self.df = df
        #print(f'self.df {self.df}')
        self.env = env
        vars = {"SL": Real(bounds=(0.0001, 0.0150)),
                "enter_at": Real(bounds=(0.001, 1.000)),
                "close_at": Real(bounds=(0.001, 1.000)),
                "type": Integer(bounds=(0, 35)),
                "MAperiod": Integer(bounds=(2, 1_000)),
                "ATRperiod": Integer(bounds=(1, 1_000)),
                "ATRmulti": Real(bounds=(0.001, 15.000))}
        super().__init__(vars=vars, n_obj=1, **kwargs)
    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['SL'], X['enter_at'], X['close_at'], X['type'], X['MAperiod'], X['ATRperiod'], X['ATRmulti']]
        #print(action)
        #env = BandsStratEnv(df=self.df, max_steps=86_400, init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPP)
        #env = BandsStratEnv(df=self.df, max_steps=604_800, init_balance=1_000, fee=0.0, coin_step=0.00001)
        _, reward, _, info = self.env.step(action)
        out["F"] = array([-reward])
        #collect()

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        #self.opt = zeros(N_GEN)
        #self.idx = 0
        self.opt = []
    def notify(self, algorithm):
        avg_rew = mean(algorithm.pop.get("F"))
        if avg_rew!=inf and avg_rew!=-inf:
            self.opt.append(avg_rew)
        #print(f'avg_rew {avg_rew}')
        #print(f'self.opt {self.opt}')
        #self.opt[self.idx] = avg_rew if avg_rew<0 else 0.0
        #self.opt[self.idx] = avg_rew
        #self.idx += 1

def display_callback(callback, fname):
    plt.title("Convergence")
    plt.ylabel('Mean Reward')
    plt.xlabel('Population')
    plt.plot(-1*array(callback.opt), "--")
    plt.savefig(getcwd()+'/reports/Convergence_'+fname)
    return plt
    #plt.show()

def display_result(result, problem, fname):
    X_array = array([[entry['type'], entry['MAperiod'], entry['ATRperiod'], entry['ATRmulti'], entry['SL'], entry['enter_at'], entry['close_at']] for entry in result.pop.get("X")])
    pop_size = len(X_array)
    labels = ['type','MAperiod','ATRperiod','ATRmulti','SL','enter_at','close_at']
    bounds = array([problem.vars[name].bounds for name in labels]).T
    plot = PCP(labels=labels, bounds=bounds, n_ticks=10)
    plot.set_axis_style(color="grey", alpha=1)
    plot.add(X_array[int(pop_size*.9)+1:], linewidth=1.9, color='#a4f0ff')
    plot.add(X_array[int(pop_size*.8)+1:int(pop_size*.9)], linewidth=1.8, color='#88e7fa')
    plot.add(X_array[int(pop_size*.7)+1:int(pop_size*.8)], linewidth=1.7, color='#60d8f3')
    plot.add(X_array[int(pop_size*.6)+1:int(pop_size*.7)], linewidth=1.6, color='#33c5e8')
    plot.add(X_array[int(pop_size*.5)+1:int(pop_size*.6)], linewidth=1.5, color='#12b0da')
    plot.add(X_array[int(pop_size*.4)+1:int(pop_size*.5)], linewidth=1.4, color='#019cc8')
    plot.add(X_array[int(pop_size*.3)+1:int(pop_size*.4)], linewidth=1.3, color='#0086b4')
    plot.add(X_array[int(pop_size*.2)+1:int(pop_size*.3)], linewidth=1.2, color='#00719f')
    plot.add(X_array[int(pop_size*.1)+1:int(pop_size*.2)], linewidth=1.1, color='#005d89')
    plot.add(X_array[:int(pop_size*.1)], linewidth=1.0, color='#004a73')
    plot.add(X_array[0], linewidth=1.5, color='red')
    plot.save(getcwd()+'/reports/'+fname)
    return plot
    #plot.show()

def main():
    #df = by_DataClient(ticker='BTCFDUSD', interval='1s', futures=False, statements=True, delay=3_600)
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1s', type='spot', data='klines', delay=129_600)
    df = df.drop(columns='Opened').to_numpy()
    df = hstack((df, zeros((df.shape[0], 1))))
    df = df[-seconds_since('09-01-2023'):,:]
    print(df)
    env = BandsStratEnv(df=df, max_steps=86_400, init_balance=1_000, fee=0.0, coin_step=0.00001, slippage=SLIPP)

    pool = Pool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)

    '''
    pool = ThreadPool(CPU_CORES_COUNT)
    runner = StarmapParallelization(pool.starmap)
    '''

    problem = CustomMixedVariableProblem(env, elementwise_runner=runner)
    #algorithm = NSGA2(pop_size=100)
    #algorithm = DNSGA2(pop_size=64)
    #algorithm = MixedVariableGA(pop=10)
    algorithm = NSGA2(  pop_size=POP_SIZE,
                        sampling=MixedVariableSampling(),
                        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
                        eliminate_duplicates=MixedVariableDuplicateElimination()    )
    #algorithm = Optuna() 

    res = minimize(problem,
                   algorithm,
                   save_history=False,
                   callback=MyCallback(),
                   termination=('n_gen', N_GEN),
                   #termination=("time", "09:00:00"),
                   verbose=True)

    print('Exec time:', res.exec_time)
    print('FINAL POPULATION')
    with open(getcwd()+'/reports/pop'+str(POP_SIZE)+str(dt.today()).replace(':','-')[:-7]+'.txt', 'w') as file:
        for f,x in zip(res.pop.get("F"), res.pop.get("X")):
            vars = round(x['SL'],4), round(x['enter_at'],3), round(x['close_at'],3), x['type'], x['MAperiod'], x['ATRperiod'], round(x['ATRmulti'],3)
            file.write('F '+str(f)+' X '+str(vars)+'\n')
            print(f'F {f} X {vars}')
    #print(f'res.pop {res.pop}')
    #print(f'res.pop.get(X) {res.pop.get("X")}')
    #print(f'res.pop.get(F) {res.pop.get("F")}')
    if len(res.F)==1:
        if isinstance(res.X, dict):
            print(f'Reward: {-res.f} Variables: {round(res.X["SL"],4), round(res.X["enter_at"],3), round(res.X["close_at"],3), res.X["type"], res.X["MAperiod"], res.X["ATRperiod"], round(res.X["ATRmulti"],3)}')
            filename = f'Pop{POP_SIZE}Rew{-res.f:.0f}Vars{round(res.X["SL"],4):.4f}-{round(res.X["enter_at"],3):.3f}-{round(res.X["close_at"],3):.3f}-{res.X["type"]}-{res.X["MAperiod"]}-{res.X["ATRperiod"]}-{round(res.X["ATRmulti"],3):.3f}.png'
        else:
            print(f'Reward: {-res.f} Variables: {round(res.X[0],4),int(res.X[1]),int(res.X[2]),int(res.X[3]),round(res.X[4],3)}')
            filename = 'Pop'+str(POP_SIZE)+'Rew'+str(-res.f)+'Vars'+str(round(res.X[0],4))+str(res.X[1])+str(res.X[2])+str(res.X[3])+str(round(res.X[4],3))+'.png'
    else:
        for front, var in zip(res.F, res.X):
            print(f"Reward:", front , "Variables:", var)
            filename = 'Figure.png'
    
    plt1 = display_callback(res.algorithm.callback, filename)
    plt2 = display_result(res, problem, filename)
    #plt.subplot(plt1)
    #plt1.show()
    plt1.show()
    #plt1.clf()
    #plt2.show()
    #time(1)
    #plt2.show()

if __name__ == '__main__':
    #profiler = cProfile.Profile() 
    #profiler.enable()
    main()
    #profiler.disable()  # Zakończ profilowanie
    #profiler.print_stats(sort='tottime')