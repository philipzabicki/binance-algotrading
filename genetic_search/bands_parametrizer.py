from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.bands_env import BandsOptimizeSpotEnv, BandsOptimizeFuturesEnv, BandsOptimizeSavingSpotEnv, \
    BandsOptimizeSavingFuturesEnv
from genetic_search.base import reward_from_metric


class BandsSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = BandsOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 36)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "stop_loss": Real(bounds=(0.0001, 0.0500)),
                           "take_profit": Real(bounds=(0.0001, 1.0000)),
                           "enter_at": Real(bounds=(0.001, 1.000)),
                           "close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['stop_loss'], X['take_profit'],
                  X['enter_at'], X['close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class BandsFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = BandsOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"position_ratio": Integer(bounds=(1, 100)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 36)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "leverage": Integer(bounds=(1, 125)),
                           "stop_loss": Real(bounds=(0.0001, 0.0150)),
                           "take_profit": Real(bounds=(0.0001, 1.0000)),
                           "long_enter_at": Real(bounds=(0.001, 1.000)),
                           "long_close_at": Real(bounds=(0.001, 1.000)),
                           "short_enter_at": Real(bounds=(0.001, 1.000)),
                           "short_close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['position_ratio'], X['stop_loss'], X['take_profit'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class BandsSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = BandsOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"save_ratio": Integer(bounds=(1, 100)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 36)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "stop_loss": Real(bounds=(0.0001, 0.0500)),
                           "take_profit": Real(bounds=(0.0001, 1.0000)),
                           "enter_at": Real(bounds=(0.001, 1.000)),
                           "close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['save_ratio'], X['stop_loss'], X['take_profit'],
                  X['enter_at'], X['close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period']]
        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class BandsSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = BandsOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"position_ratio": Integer(bounds=(1, 100)),
                           "save_ratio": Integer(bounds=(1, 100)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 36)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "leverage": Integer(bounds=(1, 125)),
                           "stop_loss": Real(bounds=(0.0001, 0.0150)),
                           "take_profit": Real(bounds=(0.0001, 1.0000)),
                           "long_enter_at": Real(bounds=(0.001, 1.000)),
                           "long_close_at": Real(bounds=(0.001, 1.000)),
                           "short_enter_at": Real(bounds=(0.001, 1.000)),
                           "short_close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['position_ratio'], X['save_ratio'],
                  X['atr_period'], X['ma_type'],
                  X['ma_period'], X['leverage'],
                  X['stop_loss'], X['take_profit'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['atr_multi']]

        if self.n_evals > 1:
            rews = array([-1 * self.env.step(action)[1] for _ in range(self.n_evals)])
            out["F"] = array([reward_from_metric(rews, self.n_evals, self.metric)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
