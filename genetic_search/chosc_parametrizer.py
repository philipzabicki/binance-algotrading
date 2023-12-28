from numpy import array
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

from enviroments.chaikinosc_env import ChaikinOscillatorStratSpotEnv, ChaikinOscillatorStratFuturesEnv


class ChaikinOscillatorMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, env_kwargs, **kwargs):
        self.env = ChaikinOscillatorStratSpotEnv(df=df, **env_kwargs)
        macd_variables = {"stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 26)),
                          "slow_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['stop_loss'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        _, reward, _, _, _ = self.env.step(action)
        # print(f'_evaluate() reward:{reward}')
        out["F"] = array([-reward])


class ChaikinOscillatorFuturesMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, df_mark, env_kwargs, **kwargs):
        self.env = ChaikinOscillatorStratFuturesEnv(df=df, df_mark=df_mark, **env_kwargs)
        macd_variables = {"position_ratio": Real(bounds=(0.01, 1.00)),
                          "stop_loss": Real(bounds=(0.0001, 0.0150)),
                          "leverage": Integer(bounds=(1, 125)),
                          "fast_period": Integer(bounds=(2, 1_000)),
                          "slow_period": Integer(bounds=(2, 1_000)),
                          "fast_ma_type": Integer(bounds=(0, 26)),
                          "slow_ma_type": Integer(bounds=(0, 26))}
        super().__init__(vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        action = [X['position_ratio'], X['stop_loss'],  X['leverage'],
                  X['fast_period'], X['slow_period'],
                  X['fast_ma_type'], X['slow_ma_type']]
        _, reward, _, _, _ = self.env.step(action)
        # print(f'_evaluate() reward:{reward}')
        out["F"] = array([-reward])
