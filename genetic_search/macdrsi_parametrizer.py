# from enviroments.macdrsi_env import MACDRSIStratSpotEnv, MACDRSIStratFuturesEnv
from numpy import array, median
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer


class MACDRSIMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, **kwargs):
        # self.env = MACDRSIStratSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at1": Real(bounds=(0.001, 1.000)),
                          "close_at1": Real(bounds=(0.001, 1.000)),
                          "enter_at2": Real(bounds=(0.001, 1.000)),
                          "close_at2": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "rsi_period": Integer(bounds=(2, 100))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['enter_at1'], X['close_at1'], X['enter_at2'], X['close_at2'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
                  X['rsi_period']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class MACDRSIFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, **kwargs):
        # self.env = MACDRSIStratFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "enter_at1": Real(bounds=(0.001, 1.000)),
                          "close_at1": Real(bounds=(0.001, 1.000)),
                          "enter_at2": Real(bounds=(0.001, 1.000)),
                          "close_at2": Real(bounds=(0.001, 1.000)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "signal_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 37)),
                          "slow_ma_type": Integer(bounds=(0, 37)),
                          "signal_ma_type": Integer(bounds=(0, 25)),
                          "rsi_period": Integer(bounds=(2, 100)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'], X['enter_at1'], X['close_at1'], X['enter_at2'], X['close_at2'],
                  X['fast_period'], X['slow_period'], X['signal_period'],
                  X['fast_ma_type'], X['slow_ma_type'], X['signal_ma_type'],
                  X['rsi_period'], X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
