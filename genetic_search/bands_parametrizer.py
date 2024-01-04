from numpy import array, mean, median
from scipy.stats import hmean
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.bands_env import BandsStratSpotEnv, BandsStratFuturesEnv


class BandsMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, **kwargs):
        self.env = BandsStratSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
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
            out["F"] = array([mean(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class BandsFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, **kwargs):
        self.env = BandsStratFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at": Real(bounds=(0.001, 1.000)),
                          "close_at": Real(bounds=(0.001, 1.000)),
                          "atr_multi": Real(bounds=(0.001, 15.000)),
                          "atr_period": Integer(bounds=(2, 1_000)),
                          "ma_type": Integer(bounds=(0, 37)),
                          "ma_period": Integer(bounds=(2, 1_000)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        action = [X['position_ratio'], X['stop_loss'],
                  X['enter_at'], X['close_at'],
                  X['atr_multi'], X['atr_period'],
                  X['ma_type'], X['ma_period'],
                  X['leverage']]

        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
