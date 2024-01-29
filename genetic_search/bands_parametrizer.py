from numpy import array, median, mean
from math import copysign
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.bands_env import BandsOptimizeSpotEnv, BandsOptimizeFuturesEnv, BandsOptimizeSavingSpotEnv, \
    BandsOptimizeSavingFuturesEnv


class BandsSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = BandsOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"stop_loss": Real(bounds=(0.0001, 0.0500)),
                           "enter_at": Real(bounds=(0.001, 1.000)),
                           "close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 37)),
                           "ma_period": Integer(bounds=(2, 1_000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['stop_loss'], X['enter_at'], X['close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period']]

        if self.n_evals > 1:
            rews = [-self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'rews mean {mean(rews)}')
            if self.metric == 'mixed':
                _median = median(rews)
                _mean = mean(rews)
                rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                out["F"] = array([rew])
            elif self.metric == 'median':
                out["F"] = array([median(rews)])
            elif self.metric == 'mean':
                out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class BandsFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = BandsOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                           "stop_loss": Real(bounds=(0.0001, 0.0150)),
                           "long_enter_at": Real(bounds=(0.001, 1.000)),
                           "long_close_at": Real(bounds=(0.001, 1.000)),
                           "short_enter_at": Real(bounds=(0.001, 1.000)),
                           "short_close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 37)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['position_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period'],
                  X['leverage']]

        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.metric == 'mixed':
                _median = median(rews)
                _mean = mean(rews)
                rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                out["F"] = array([rew])
            elif self.metric == 'median':
                out["F"] = array([median(rews)])
            elif self.metric == 'mean':
                out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class BandsSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = BandsOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"save_ratio": Real(bounds=(0.0, 1.0)),
                           "stop_loss": Real(bounds=(0.0001, 0.0500)),
                           "enter_at": Real(bounds=(0.001, 1.000)),
                           "close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 37)),
                           "ma_period": Integer(bounds=(2, 1_000))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['save_ratio'], X['stop_loss'],
                  X['enter_at'], X['close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period']]

        if self.n_evals > 1:
            rews = [-self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'rews mean {mean(rews)}')
            if self.metric == 'mixed':
                _median = median(rews)
                _mean = mean(rews)
                rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                out["F"] = array([rew])
            elif self.metric == 'median':
                out["F"] = array([median(rews)])
            elif self.metric == 'mean':
                out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class BandsSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = BandsOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        bands_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                           "save_ratio": Real(bounds=(0.0, 1.0)),
                           "stop_loss": Real(bounds=(0.0001, 0.0150)),
                           "long_enter_at": Real(bounds=(0.001, 1.000)),
                           "long_close_at": Real(bounds=(0.001, 1.000)),
                           "short_enter_at": Real(bounds=(0.001, 1.000)),
                           "short_close_at": Real(bounds=(0.001, 1.000)),
                           "atr_multi": Real(bounds=(0.001, 15.000)),
                           "atr_period": Integer(bounds=(2, 1_000)),
                           "ma_type": Integer(bounds=(0, 37)),
                           "ma_period": Integer(bounds=(2, 1_000)),
                           "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['position_ratio'], X['save_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period'],
                  X['leverage']]

        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.metric == 'mixed':
                _median = median(rews)
                _mean = mean(rews)
                rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                out["F"] = array([rew])
            elif self.metric == 'median':
                out["F"] = array([median(rews)])
            elif self.metric == 'mean':
                out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
