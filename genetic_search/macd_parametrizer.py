from numpy import array, median, mean
from math import copysign
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.macd_env import MACDOptimizeSpotEnv, MACDOptimizeFuturesEnv, MACDOptimizeSavingFuturesEnv, \
    MACDOptimizeSavingSpotEnv


class MACDSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = MACDOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['enter_at'], X['close_at'], X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
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


class MACDFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = MACDOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "long_enter_at": Real(bounds=(0.001, 1.000)),
                          "long_close_at": Real(bounds=(0.001, 1.000)),
                          "short_enter_at": Real(bounds=(0.001, 1.000)),
                          "short_close_at": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 26)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
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
class MACDSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = MACDOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"save_ratio": Real(bounds=(0.0, 1.0)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'],
                  X['enter_at'], X['close_at'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type']]
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


class MACDSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='mixed', **kwargs):
        self.env = MACDOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "save_ratio": Real(bounds=(0.0, 1.0)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "long_enter_at": Real(bounds=(0.001, 1.000)),
                          "long_close_at": Real(bounds=(0.001, 1.000)),
                          "short_enter_at": Real(bounds=(0.001, 1.000)),
                          "short_close_at": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 26)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.metric == 'mixed':
                _median = median(rews)
                _mean = mean(rews)
                rew = (_median+_mean)/2 if (_median < 0) or (_mean < 0) else _median * _mean
                out["F"] = array([rew])
            elif self.metric == 'median':
                out["F"] = array([median(rews)])
            elif self.metric == 'mean':
                out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
