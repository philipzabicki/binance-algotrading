from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer

from enviroments.signal_env import SignalExecuteSpotEnv


class SpotActionVariablesProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, **kwargs):
        self.env = SignalExecuteSpotEnv(df=df, **env_kwargs)
        super().__init__(
            n_var=df.shape[0],
            n_obj=1,
            n_constr=0,
            xl=-1,
            xu=1,
            vtype=int,
            type_var=Integer
        )

    def _evaluate(self, X, out, *args, **kwargs):
        self.env.reset()
        # print(f'X {X}')
        self.env.signals = X
        out["F"] = array([-self.env()[1]])
