from numpy import array, median
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.zajeciowy_env import ChaikinOscillatorStratSpotEnv, ChaikinOscillatorStratFuturesEnv


class ChaikinOscillatorMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, n_evals=1, **kwargs):
        self.env = ChaikinOscillatorStratSpotEnv(df=df, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"fast_period": Integer(bounds=(2, 100)),
                          "slow_period": Integer(bounds=(2, 100)),
                          "fast_ma_type": Integer(bounds=(0, 26)),
                          "slow_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])


class ChaikinOscillatorFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, n_evals=1, **kwargs):
        self.env = ChaikinOscillatorStratFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        self.n_evals = n_evals
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 26)),
                          "slow_ma_type": Integer(bounds=(0, 26)),
                          "leverage": Integer(bounds=(1, 125))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type'],
                  X['leverage']]
        if self.n_evals > 1:
            rews = [-1 * self.env.step(action)[1] for _ in range(self.n_evals)]
            # print(f'median_of{self.n_evals}_reward: {median(rews)}')
            out["F"] = array([median(rews)])
        else:
            out["F"] = array([-self.env.step(action)[1]])
