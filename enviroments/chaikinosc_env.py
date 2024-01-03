from gym import spaces, Env
from numpy import array, float64, inf
from talib import AD

from utils.ta_tools import custom_ChaikinOscillator, ChaikinOscillator_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


class ChaikinOscillatorExecuteSpotEnv(SignalExecuteSpotEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adl = AD(self.df[:, 1], self.df[:, 2], self.df[:, 3], self.df[:, 4])

    def reset(self, *args, stop_loss=None,
              fast_period=3, slow_period=10,
              fast_ma_type=0, slow_ma_type=0, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        _max_period = max(self.fast_period, self.slow_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        # print(self.df[self.start_step:self.end_step, :5])
        chaikin_oscillator = custom_ChaikinOscillator(self.adl[prev_values:self.end_step, ],
                                                      fast_ma_type=fast_ma_type, fast_period=fast_period,
                                                      slow_ma_type=slow_ma_type, slow_period=slow_period)
        self.signals = ChaikinOscillator_signal(chaikin_oscillator[self.start_step - prev_values:])
        # print(f'len sig {len(self.signals)}')
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' fast_period={self.fast_period}, slow_period={self.slow_period}')
            print(f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}')
        # Fix for multiprocessing to not show other episodes after one reaches this condition
        if self.balance >= 1_000_000:
            self.verbose = False


class ChaikinOscillatorStratSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = ChaikinOscillatorExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0, 0, 2, 2]
        action_upper = [0.0500, 26, 26, 10_000, 10_000]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, fast_period=12, slow_period=26,
              fast_ma_type=1, slow_ma_type=1):
        return self.exec_env.reset(stop_loss=stop_loss, fast_period=fast_period, slow_period=slow_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type)

    def step(self, action):
        self.reset(stop_loss=action[0],
                   fast_period=int(action[1]), slow_period=int(action[2]),
                   fast_ma_type=int(action[3]), slow_ma_type=int(action[4]))
        return self.exec_env()


########################################################################################################################
# FUTURES
class ChaikinOscillatorExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adl = AD(self.df[:, 1], self.df[:, 2], self.df[:, 3], self.df[:, 4])
        # print(f'futures self.adl {self.adl}')
        # print(f'len(self.adl) {len(self.adl)}')

    def reset(self, *args, position_ratio=1.0,
              stop_loss=None, leverage=5,
              fast_period=12, slow_period=26,
              fast_ma_type=0, slow_ma_type=0, **kwargs):
        _ret = super().reset(*args, position_ratio=position_ratio,
                             leverage=leverage, stop_loss=stop_loss, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        _max_period = max(self.fast_period, self.slow_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        # print(self.df[self.start_step:self.end_step, :5])
        chaikin_oscillator = custom_ChaikinOscillator(self.adl[prev_values:self.end_step, ],
                                                      fast_ma_type=fast_ma_type, fast_period=fast_period,
                                                      slow_ma_type=slow_ma_type, slow_period=slow_period)
        self.signals = ChaikinOscillator_signal(chaikin_oscillator[self.start_step - prev_values:])
        # print(f'start_step {self.start_step} end_step {self.end_step} _max_period {_max_period} prev_values {prev_values}')
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' fast_period={self.fast_period}, slow_period={self.slow_period}')
            print(f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}')
        if self.balance >= 1_000_000:
            self.verbose = False


class ChaikinOscillatorStratFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = ChaikinOscillatorExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.01, 0.0001, 2, 2, 0, 0, 1]
        action_upper = [1.0, 0.0500, 10_000, 10_000, 26, 26, 125]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, leverage=5, stop_loss=None,
              fast_period=12, slow_period=26,
              fast_ma_type=1, slow_ma_type=1):
        return self.exec_env.reset(position_ratio=position_ratio, stop_loss=stop_loss, leverage=leverage,
                                   fast_period=fast_period, slow_period=slow_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1],
                   fast_period=int(action[2]), slow_period=int(action[3]),
                   fast_ma_type=int(action[4]), slow_ma_type=int(action[5]), leverage=int(action[6]))
        return self.exec_env()
