from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice

from enviroments.stoch_env import StochOptimizeSpotEnv, StochOptimizeFuturesEnv, StochOptimizeSavingSpotEnv, \
    StochOptimizeSavingFuturesEnv
from genetic_search.base import reward_from_metric


class StochSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = StochOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"oversold_threshold": Integer(bounds=(0, 500)),
                          "overbought_threshold": Integer(bounds=(500, 1_000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 25)),
                          "slowD_ma_type": Integer(bounds=(0, 25)),
                          "stop_loss": Real(bounds=(0.0001, 0.015)),
                          "take_profit": Real(bounds=(0.0001, 1.0000)),
                          "enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['take_profit'],
                  X['enter_at'], X['close_at'],
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class StochFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = StochOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Integer(bounds=(1, 100)),
                          "oversold_threshold": Integer(bounds=(0, 500)),
                          "overbought_threshold": Integer(bounds=(500, 1_000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 25)),
                          "slowD_ma_type": Integer(bounds=(0, 25)),
                          "leverage": Integer(bounds=(1, 125)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "take_profit": Real(bounds=(0.0001, 1.0000)),
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
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class StochSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = StochOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"save_ratio": Integer(bounds=(1, 100)),
                          "oversold_threshold": Integer(bounds=(0, 500)),
                          "overbought_threshold": Integer(bounds=(500, 1_000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 25)),
                          "slowD_ma_type": Integer(bounds=(0, 25)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "take_profit": Real(bounds=(0.0001, 1.0000)),
                          "enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'], X['take_profit'],
                  X['enter_at'], X['close_at'],
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class StochSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = StochOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Integer(bounds=(1, 100)),
                          "save_ratio": Integer(bounds=(1, 100)),
                          "oversold_threshold": Integer(bounds=(0, 500)),
                          "overbought_threshold": Integer(bounds=(500, 1_000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 25)),
                          "slowD_ma_type": Integer(bounds=(0, 25)),
                          "leverage": Integer(bounds=(1, 125)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "take_profit": Real(bounds=(0.0001, 1.0000)),
                          "long_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "long_close_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_enter_at": Choice(options=[.25, .5, .75, 1.]),
                          "short_close_at": Choice(options=[.25, .5, .75, 1.])}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'],
                  X['stop_loss'], X['take_profit'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
