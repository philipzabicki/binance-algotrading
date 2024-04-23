from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.chaikinosc_env import ChaikinOscillatorOptimizeSpotEnv, ChaikinOscillatorOptimizeFuturesEnv, \
    ChaikinOscillatorOptimizeSavingSpotEnv, ChaikinOscillatorOptimizeSavingFuturesEnv
from genetic_search.base import reward_from_metric


class ChaikinOscillatorSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "take_profit": Real(bounds=(0.0001, 1.0000))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['take_profit'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class ChaikinOscillatorFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"position_ratio": Integer(bounds=(1, 100)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25)),
                             "leverage": Integer(bounds=(1, 125)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "take_profit": Real(bounds=(0.0001, 1.0000))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'], X['take_profit'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class ChaikinOscillatorSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"save_ratio": Integer(bounds=(1, 100)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "take_profit": Real(bounds=(0.0001, 1.0000))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'], X['take_profit'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class ChaikinOscillatorSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"position_ratio": Integer(bounds=(1, 100)),
                             "save_ratio": Integer(bounds=(1, 100)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25)),
                             "leverage": Integer(bounds=(1, 125)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "take_profit": Real(bounds=(0.0001, 1.0000))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'],
                  X['stop_loss'], X['take_profit'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
