from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice

from enviroments.macd_env import MACDOptimizeSpotEnv, MACDOptimizeFuturesEnv, MACDOptimizeSavingFuturesEnv, \
    MACDOptimizeSavingSpotEnv
from genetic_search.base import reward_from_metric


# TODO: add position_ratio parameter for spot ones
class MACDSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = MACDOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"fast_period": Integer(bounds=(2, 250)),
                          "slow_period": Integer(bounds=(2, 250)),
                          "signal_period": Integer(bounds=(2, 250)),
                          "fast_ma_type": Integer(bounds=(0, 36)),
                          "slow_ma_type": Integer(bounds=(0, 36)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "stop_loss": Real(bounds=(.0001, .10)),
                          "take_profit": Real(bounds=(.0001, .10)),
                          "enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['take_profit'], X['enter_at'], X['close_at'], X['fast_period'], X['slow_period'],
                  X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class MACDFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = MACDOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Integer(bounds=(1, 100)),
                          "fast_period": Integer(bounds=(2, 250)),
                          "slow_period": Integer(bounds=(2, 250)),
                          "signal_period": Integer(bounds=(2, 250)),
                          "fast_ma_type": Integer(bounds=(0, 36)),
                          "slow_ma_type": Integer(bounds=(0, 36)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "leverage": Integer(bounds=(1, 125)),
                          "stop_loss": Real(bounds=(.0001, .10)),
                          "take_profit": Real(bounds=(.0001, .10)),
                          "long_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "long_close_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'], X['take_profit'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class MACDSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = MACDOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"save_ratio": Integer(bounds=(1, 100)),
                          "fast_period": Integer(bounds=(2, 250)),
                          "slow_period": Integer(bounds=(2, 250)),
                          "signal_period": Integer(bounds=(2, 250)),
                          "fast_ma_type": Integer(bounds=(0, 36)),
                          "slow_ma_type": Integer(bounds=(0, 36)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "stop_loss": Real(bounds=(.0001, .10)),
                          "take_profit": Real(bounds=(.0001, .10)),
                          "enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'], X['take_profit'],
                  X['enter_at'], X['close_at'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class MACDSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = MACDOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Integer(bounds=(1, 100)),
                          "save_ratio": Integer(bounds=(1, 100)),
                          "fast_period": Integer(bounds=(2, 250)),
                          "slow_period": Integer(bounds=(2, 250)),
                          "signal_period": Integer(bounds=(2, 250)),
                          "fast_ma_type": Integer(bounds=(0, 36)),
                          "slow_ma_type": Integer(bounds=(0, 36)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "leverage": Integer(bounds=(1, 125)),
                          "stop_loss": Real(bounds=(.0001, .10)),
                          "take_profit": Real(bounds=(.0001, .10)),
                          "long_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "long_close_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
                  X['leverage'], X['stop_loss'], X['take_profit'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
