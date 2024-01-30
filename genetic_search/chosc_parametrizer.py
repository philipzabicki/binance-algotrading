from numpy import array, median, mean, percentile
from math import copysign
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.chaikinosc_env import ChaikinOscillatorOptimizeSpotEnv, ChaikinOscillatorOptimizeFuturesEnv, ChaikinOscillatorOptimizeSavingSpotEnv, ChaikinOscillatorOptimizeSavingFuturesEnv


class ChaikinOscillatorSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 26)),
                             "slow_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.n_evals > 1:
                rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
                # print(f'median_of{self.n_evals}_reward: {median(rews)}')
                if self.metric == 'mixed':
                    _median = median(rews)
                    _mean = mean(rews)
                    rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                    out["F"] = array([rew])
                elif self.metric == '10perc_x_mean':
                    out["F"] = array([percentile(rews, 10) * mean(rews)])
                elif self.metric == 'median':
                    out["F"] = array([median(rews)])
                elif self.metric == 'mean':
                    out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class ChaikinOscillatorFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 26)),
                             "slow_ma_type": Integer(bounds=(0, 26)),
                             "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.n_evals > 1:
                rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
                # print(f'median_of{self.n_evals}_reward: {median(rews)}')
                if self.metric == 'mixed':
                    _median = median(rews)
                    _mean = mean(rews)
                    rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                    out["F"] = array([rew])
                elif self.metric == '10perc_x_mean':
                    out["F"] = array([percentile(rews, 10) * mean(rews)])
                elif self.metric == 'median':
                    out["F"] = array([median(rews)])
                elif self.metric == 'mean':
                    out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class ChaikinOscillatorSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"save_ratio": Real(bounds=(0.0, 1.0)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 26)),
                             "slow_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.n_evals > 1:
                rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
                # print(f'median_of{self.n_evals}_reward: {median(rews)}')
                if self.metric == 'mixed':
                    _median = median(rews)
                    _mean = mean(rews)
                    rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                    out["F"] = array([rew])
                elif self.metric == '10perc_x_mean':
                    out["F"] = array([percentile(rews, 10) * mean(rews)])
                elif self.metric == 'median':
                    out["F"] = array([median(rews)])
                elif self.metric == 'mean':
                    out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class ChaikinOscillatorSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, metric='median', **kwargs):
        self.env = ChaikinOscillatorOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        self.metric = metric
        chaikin_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                             "save_ratio": Real(bounds=(0.0, 1.0)),
                             "stop_loss": Real(bounds=(0.0001, 0.0150)),
                             "fast_period": Integer(bounds=(2, 1_000)),
                             "slow_period": Integer(bounds=(2, 1_000)),
                             "fast_ma_type": Integer(bounds=(0, 26)),
                             "slow_ma_type": Integer(bounds=(0, 26)),
                             "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'], X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            if self.n_evals > 1:
                rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
                # print(f'median_of{self.n_evals}_reward: {median(rews)}')
                if self.metric == 'mixed':
                    _median = median(rews)
                    _mean = mean(rews)
                    rew = (_median + _mean) / 2 if (_median < 0) or (_mean < 0) else _median * _mean
                    out["F"] = array([rew])
                elif self.metric == '10perc_x_mean':
                    out["F"] = array([percentile(rews, 10) * mean(rews)])
                elif self.metric == 'median':
                    out["F"] = array([median(rews)])
                elif self.metric == 'mean':
                    out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
