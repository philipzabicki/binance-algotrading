from warnings import warn

from gym import spaces, Env
from numpy import array, float64, inf

from definitions import ADDITIONAL_DATA_BY_MA
from utils.ta_tools import custom_StochasticOscillator, StochasticOscillator_signal
from .signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv


########################################################################################################################
# EXECUTING ENVIRONMENTS
class _StochExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, take_profit=None, save_ratio=None, enter_at=1.0, close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0, **kwargs):
        self.fastK_period = fastK_period
        self.slowK_period = slowK_period
        self.slowD_period = slowD_period
        self.slowK_ma_type = slowK_ma_type
        self.slowD_ma_type = slowD_ma_type
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        # Some MAs need as much additional previous data as 25 times period length to return repetitive values.
        # E.g. MA15 from 100 datapoints may be not the same as MA15 from 1000 datapoints, but we ignore that for now.
        _max_period = self.fastK_period + self.slowK_period * ADDITIONAL_DATA_BY_MA[slowK_ma_type] + self.slowD_period * \
                      ADDITIONAL_DATA_BY_MA[slowD_ma_type]
        if _max_period > self.total_steps:
            warn(
                f'Previous data required for consistent MAs calculation is larger than whole df. ({_max_period} vs {self.total_steps})')
        _ret = super().reset(*args, offset=_max_period, stop_loss=stop_loss, take_profit=take_profit, save_ratio=save_ratio,
                             enter_at=enter_at, close_at=close_at, **kwargs)
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        slowK, slowD = custom_StochasticOscillator(self.df[prev_values:self.end_step, :5],
                                                   fastK_period=fastK_period,
                                                   slowK_period=slowK_period,
                                                   slowD_period=slowD_period,
                                                   slowK_ma_type=slowK_ma_type,
                                                   slowD_ma_type=slowD_ma_type)
        self.signals = StochasticOscillator_signal(slowK[self.start_step - prev_values:],
                                                   slowD[self.start_step - prev_values:],
                                                   oversold_threshold=oversold_threshold,
                                                   overbought_threshold=overbought_threshold)
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fastK_period={self.fastK_period}, slowK_period={self.slowK_period}, slowD_period={self.slowD_period}')
            print(f' slowK_ma_type={self.slowK_ma_type}, slowD_ma_type={self.slowD_ma_type}')
            print(f' oversold_threshold={self.oversold_threshold}, overbought_threshold={self.overbought_threshold}')
        if self.balance >= 1_000_000:
            self.verbose = False


class _StochExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None, take_profit=None, save_ratio=None,
              long_enter_at=1.0, long_close_at=1.0, short_enter_at=1.0, short_close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0, leverage=5, **kwargs):
        self.fastK_period = fastK_period
        self.slowK_period = slowK_period
        self.slowD_period = slowD_period
        self.slowK_ma_type = slowK_ma_type
        self.slowD_ma_type = slowD_ma_type
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        # Some MAs need as much additional previous data as 25 times period length to return repetitive values.
        # E.g. MA15 from 100 datapoints may be not the same as MA15 from 1000 datapoints, but we ignore that for now.
        _max_period = self.fastK_period + self.slowK_period * ADDITIONAL_DATA_BY_MA[slowK_ma_type] + self.slowD_period * \
                      ADDITIONAL_DATA_BY_MA[slowD_ma_type]
        if _max_period > self.total_steps:
            warn(
                f'Previous data required for consistent MAs calculation is larger than whole df. ({_max_period} vs {self.total_steps})')
        _ret = super().reset(*args, offset=_max_period, position_ratio=position_ratio, save_ratio=save_ratio,
                             leverage=leverage, stop_loss=stop_loss, take_profit=take_profit,
                             long_enter_at=long_enter_at, long_close_at=long_close_at,
                             short_enter_at=short_enter_at, short_close_at=short_close_at, **kwargs)
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        slowK, slowD = custom_StochasticOscillator(self.df[prev_values:self.end_step, :5],
                                                   fastK_period=fastK_period,
                                                   slowK_period=slowK_period,
                                                   slowD_period=slowD_period,
                                                   slowK_ma_type=slowK_ma_type,
                                                   slowD_ma_type=slowD_ma_type)
        # print(slowK)
        # print(slowD)
        self.signals = StochasticOscillator_signal(slowK[self.start_step - prev_values:],
                                                   slowD[self.start_step - prev_values:],
                                                   oversold_threshold=oversold_threshold,
                                                   overbought_threshold=overbought_threshold)
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fastK_period={self.fastK_period}, slowK_period={self.slowK_period}, slowD_period={self.slowD_period}')
            print(f' slowK_ma_type={self.slowK_ma_type}, slowD_ma_type={self.slowD_ma_type}')
            print(f' oversold_threshold={self.oversold_threshold}, overbought_threshold={self.overbought_threshold}')
        if self.balance >= 1_000_000:
            self.verbose = False


########################################################################################################################
# OPTIMIZE ENVIRONMENTS
class StochOptimizeSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _StochExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.0001, 0.001, 0.001, 0, 50, 2, 2, 2, 0, 0]
        action_upper = [0.0500, 1.0000, 1.000, 1.000, 50, 100, 10_000, 10_000, 10_000, 26, 26]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, take_profit=None, enter_at=1.0, close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0):
        return self.exec_env.reset(stop_loss=stop_loss, take_profit=take_profit, enter_at=enter_at, close_at=close_at,
                                   fastK_period=fastK_period, slowK_period=slowK_period, slowD_period=slowD_period,
                                   slowK_ma_type=slowK_ma_type, slowD_ma_type=slowD_ma_type,
                                   oversold_threshold=oversold_threshold, overbought_threshold=overbought_threshold)

    def step(self, action):
        self.reset(stop_loss=action[0], take_profit=action[1], enter_at=action[2], close_at=action[3],
                   oversold_threshold=action[4], overbought_threshold=action[5],
                   fastK_period=int(action[6]), slowK_period=int(action[7]), slowD_period=int(action[8]),
                   slowK_ma_type=int(action[9]), slowD_ma_type=int(action[10]))
        return self.exec_env()


class StochOptimizeFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _StochExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.01, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0, 50, 2, 2, 2, 0, 0, 1]
        action_upper = [1.00, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 50, 100, 10_000, 10_000, 10_000, 26, 26, 125]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, leverage=5,
              stop_loss=None, take_profit=None,
              long_enter_at=1.0, long_close_at=1.0,
              short_enter_at=1.0, short_close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0):
        return self.exec_env.reset(position_ratio=position_ratio, stop_loss=stop_loss, take_profit=take_profit,
                                   long_enter_at=long_enter_at, long_close_at=long_close_at,
                                   short_enter_at=short_enter_at, short_close_at=short_close_at,
                                   fastK_period=fastK_period, slowK_period=slowK_period, slowD_period=slowD_period,
                                   slowK_ma_type=slowK_ma_type, slowD_ma_type=slowD_ma_type,
                                   oversold_threshold=oversold_threshold, overbought_threshold=overbought_threshold,
                                   leverage=leverage)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1], take_profit=action[2],
                   long_enter_at=action[3], long_close_at=action[4],
                   short_enter_at=action[5], short_close_at=action[6],
                   oversold_threshold=action[7], overbought_threshold=action[8],
                   fastK_period=int(action[9]), slowK_period=int(action[10]), slowD_period=int(action[11]),
                   slowK_ma_type=int(action[12]), slowD_ma_type=int(action[13]),
                   leverage=int(action[14]))
        return self.exec_env()


########################################################################################################################
# OPTIMIZE ENVIRONMENTS WITH SAVING BALANCE PARAMETER
class StochOptimizeSavingSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _StochExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.000, 0.0001, 0.0001, 0.001, 0.001, 0, 50, 2, 2, 2, 0, 0]
        action_upper = [1.000, 0.0500, 1.0000, 1.000, 1.000, 50, 100, 10_000, 10_000, 10_000, 26, 26]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, take_profit=None, save_ratio=None,
              enter_at=1.0, close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0):
        return self.exec_env.reset(save_ratio=save_ratio, stop_loss=stop_loss, take_profit=take_profit,
                                   enter_at=enter_at, close_at=close_at,
                                   fastK_period=fastK_period, slowK_period=slowK_period, slowD_period=slowD_period,
                                   slowK_ma_type=slowK_ma_type, slowD_ma_type=slowD_ma_type,
                                   oversold_threshold=oversold_threshold, overbought_threshold=overbought_threshold)

    def step(self, action):
        self.reset(save_ratio=action[0], stop_loss=action[1], take_profit=action[2],
                   enter_at=action[3], close_at=action[4],
                   oversold_threshold=action[5], overbought_threshold=action[6],
                   fastK_period=int(action[7]), slowK_period=int(action[8]), slowD_period=int(action[9]),
                   slowK_ma_type=int(action[10]), slowD_ma_type=int(action[11]))
        return self.exec_env()


class StochOptimizeSavingFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _StochExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.01, 0.000, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0, 50, 2, 2, 2, 0, 0, 1]
        action_upper = [1.00, 1.000, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 50, 100, 10_000, 10_000, 10_000, 26,
                        26, 125]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, leverage=5,
              stop_loss=None, take_profit=None, save_ratio=None,
              long_enter_at=1.0, long_close_at=1.0,
              short_enter_at=1.0, short_close_at=1.0,
              fastK_period=14, slowK_period=1, slowD_period=3,
              slowK_ma_type=0, slowD_ma_type=0,
              oversold_threshold=20.0, overbought_threshold=80.0):
        return self.exec_env.reset(position_ratio=position_ratio, save_ratio=save_ratio,
                                   stop_loss=stop_loss, take_profit=take_profit,
                                   long_enter_at=long_enter_at, long_close_at=long_close_at,
                                   short_enter_at=short_enter_at, short_close_at=short_close_at,
                                   fastK_period=fastK_period, slowK_period=slowK_period, slowD_period=slowD_period,
                                   slowK_ma_type=slowK_ma_type, slowD_ma_type=slowD_ma_type,
                                   oversold_threshold=oversold_threshold, overbought_threshold=overbought_threshold,
                                   leverage=leverage)

    def step(self, action):
        self.reset(position_ratio=action[0], save_ratio=action[1],
                   stop_loss=action[2], take_profit=action[3],
                   long_enter_at=action[4], long_close_at=action[5],
                   short_enter_at=action[6], short_close_at=action[7],
                   oversold_threshold=action[8], overbought_threshold=action[9],
                   fastK_period=int(action[10]), slowK_period=int(action[11]), slowD_period=int(action[12]),
                   slowK_ma_type=int(action[13]), slowD_ma_type=int(action[14]),
                   leverage=int(action[15]))
        return self.exec_env()
