from numpy import array, median
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.stoch_env import StochOptimizeSpotEnv, StochOptimizeFuturesEnv, StochOptimizeSavingSpotEnv, StochOptimizeSavingFuturesEnv


class StochSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, **kwargs):
        self.env = StochOptimizeSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"stop_loss": Real(bounds=(0.0001, 0.015)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 26)),
                          "slowD_ma_type": Integer(bounds=(0, 26)),
                          "oversold_threshold": Real(bounds=(0, 50)),
                          "overbought_threshold": Real(bounds=(50, 100))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['enter_at'], X['close_at'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['oversold_threshold'], X['overbought_threshold']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class StochFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, **kwargs):
        self.env = StochOptimizeFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "long_enter_at": Real(bounds=(0.001, 1.000)),
                          "long_close_at": Real(bounds=(0.001, 1.000)),
                          "short_enter_at": Real(bounds=(0.001, 1.000)),
                          "short_close_at": Real(bounds=(0.001, 1.000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 26)),
                          "slowD_ma_type": Integer(bounds=(0, 26)),
                          "oversold_threshold": Real(bounds=(0, 50)),
                          "overbought_threshold": Real(bounds=(50, 100)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


########################################################################################################################
# SAVING ONES
class StochSavingSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, **kwargs):
        self.env = StochOptimizeSavingSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"save_ratio": Real(bounds=(0.0, 1.0)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 26)),
                          "slowD_ma_type": Integer(bounds=(0, 26)),
                          "oversold_threshold": Real(bounds=(0, 50)),
                          "overbought_threshold": Real(bounds=(50, 100))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['save_ratio'], X['stop_loss'],
                  X['enter_at'], X['close_at'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['oversold_threshold'], X['overbought_threshold']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class StochSavingFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, **kwargs):
        self.env = StochOptimizeSavingFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "save_ratio": Real(bounds=(0.0, 1.0)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "long_enter_at": Real(bounds=(0.001, 1.000)),
                          "long_close_at": Real(bounds=(0.001, 1.000)),
                          "short_enter_at": Real(bounds=(0.001, 1.000)),
                          "short_close_at": Real(bounds=(0.001, 1.000)),
                          "fastK_period": Integer(bounds=(2, 1_000)),
                          "slowK_period": Integer(bounds=(2, 1_000)),
                          "slowD_period": Integer(bounds=(2, 1_000)),
                          "slowK_ma_type": Integer(bounds=(0, 26)),
                          "slowD_ma_type": Integer(bounds=(0, 26)),
                          "oversold_threshold": Real(bounds=(0, 50)),
                          "overbought_threshold": Real(bounds=(50, 100)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['save_ratio'], X['stop_loss'],
                  X['long_enter_at'], X['long_close_at'],
                  X['short_enter_at'], X['short_close_at'],
                  X['fastK_period'], X['slowK_period'], X['slowD_period'],
                  X['slowK_ma_type'], X['slowD_ma_type'],
                  X['oversold_threshold'], X['overbought_threshold'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])