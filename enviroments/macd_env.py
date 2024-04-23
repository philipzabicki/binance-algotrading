from warnings import warn

from gym import spaces, Env
from numpy import array, float64, inf

from definitions import ADDITIONAL_DATA_BY_OHLCV_MA, ADDITIONAL_DATA_BY_MA
from enviroments.signal_env import SignalExecuteSpotEnv, SignalExecuteFuturesEnv
from utils.ta_tools import custom_MACD, MACD_cross_signal


# TODO: Check gym standards for action and observation spaces
########################################################################################################################
# EXECUTING ENVIRONMENTS
class _MACDExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, take_profit=None, save_ratio=None, enter_at=1.0, close_at=1.0,
              fast_period=12, slow_period=25, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, **kwargs):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        # Some MAs need as much additional previous data as 25 times period length to return repetitive values.
        # E.g. MA15 from 100 datapoints may be not the same as MA15 from 1000 datapoints, but we ignore that for now.
        _max_period = max(self.fast_period * ADDITIONAL_DATA_BY_OHLCV_MA[fast_ma_type],
                          self.slow_period * ADDITIONAL_DATA_BY_OHLCV_MA[slow_ma_type]) + self.signal_period * \
                      ADDITIONAL_DATA_BY_MA[signal_ma_type]
        _ret = super().reset(*args, offset=_max_period, stop_loss=stop_loss, take_profit=take_profit,
                             save_ratio=save_ratio,
                             enter_at=enter_at, close_at=close_at, **kwargs)
        if self.start_step > _max_period:
            prev_values = self.start_step - _max_period
        else:
            prev_values = 0
            warn(
                f'Previous data required for consistent MAs calculation is larger than previous values existing in df. ({_max_period} vs {self.start_step})')
        macd, macd_signal = custom_MACD(self.df[prev_values:self.end_step, 1:6],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.signals = MACD_cross_signal(macd[self.start_step - prev_values:],
                                         macd_signal[self.start_step - prev_values:])
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(
                f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')
        if self.balance >= 1_000_000:
            self.verbose = False


class _MACDExecuteFuturesEnv(SignalExecuteFuturesEnv):
    def reset(self, *args, position_ratio=1.0, stop_loss=None, take_profit=None, save_ratio=None,
              long_enter_at=1.0, long_close_at=1.0, short_enter_at=1.0, short_close_at=1.0,
              fast_period=12, slow_period=25, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, leverage=5, **kwargs):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        _max_period = max(self.fast_period * ADDITIONAL_DATA_BY_OHLCV_MA[fast_ma_type],
                          self.slow_period * ADDITIONAL_DATA_BY_OHLCV_MA[slow_ma_type]) + self.signal_period * \
                      ADDITIONAL_DATA_BY_MA[signal_ma_type]
        _ret = super().reset(*args, offset=_max_period, position_ratio=position_ratio, save_ratio=save_ratio,
                             leverage=leverage, stop_loss=stop_loss, take_profit=take_profit,
                             long_enter_at=long_enter_at, long_close_at=long_close_at,
                             short_enter_at=short_enter_at, short_close_at=short_close_at, **kwargs)
        if self.start_step > _max_period:
            prev_values = self.start_step - _max_period
            # warn(f'Previous data required ({_max_period} vs start step {self.start_step})')
        else:
            prev_values = 0
            warn(
                f'Previous data required for consistent MAs calculation is larger than previous values existing in df. ({_max_period} vs {self.start_step})')
        macd, macd_signal = custom_MACD(self.df[prev_values:self.end_step, 1:6],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.signals = MACD_cross_signal(macd[self.start_step - prev_values:],
                                         macd_signal[self.start_step - prev_values:])
        # print(f'start_step={self.start_step} len(signals)={len(self.signals)} len(df)={len(self.df)}')
        return _ret

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(
                f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')
        if self.balance >= 1_000_000:
            self.verbose = False


########################################################################################################################
# OPTIMIZE ENVIRONMENTS
class MACDOptimizeSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _MACDExecuteSpotEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.0001, 0.0001, 0.001, 0.001, 2, 2, 2, 0, 0, 0]
        action_upper = [0.0500, 1.0000, 1.000, 1.000, 1_000, 1_000, 1_000, 37, 37, 25]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(stop_loss=action[0], take_profit=action[1], enter_at=action[2], close_at=action[3],
                   fast_period=int(action[4]), slow_period=int(action[5]), signal_period=int(action[6]),
                   fast_ma_type=int(action[7]), slow_ma_type=int(action[8]), signal_ma_type=int(action[9]))
        return self.exec_env()


class MACDOptimizeFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _MACDExecuteFuturesEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.01, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 2, 2, 2, 0, 0, 0, 1]
        action_upper = [1.00, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 1_000, 1_000, 1_000, 37, 37, 25, 125]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(position_ratio=action[0], stop_loss=action[1], take_profit=action[2],
                   long_enter_at=action[3], long_close_at=action[4],
                   short_enter_at=action[5], short_close_at=action[6],
                   fast_period=int(action[7]), slow_period=int(action[8]), signal_period=int(action[9]),
                   fast_ma_type=int(action[10]), slow_ma_type=int(action[11]), signal_ma_type=int(action[12]),
                   leverage=int(action[13]))
        return self.exec_env()


########################################################################################################################
# OPTIMIZE ENVIRONMENTS WITH SAVING BALANCE PARAMETER
class MACDOptimizeSavingSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _MACDExecuteSpotEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.000, 0.0001, 0.0001, 0.001, 0.001, 2, 2, 2, 0, 0, 0]
        action_upper = [1.000, 0.0500, 1.0000, 1.000, 1.000, 1_000, 1_000, 1_000, 36, 36, 25]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(save_ratio=action[0], stop_loss=action[1], take_profit=action[2], enter_at=action[3],
                   close_at=action[4],
                   fast_period=int(action[5]), slow_period=int(action[6]), signal_period=int(action[7]),
                   fast_ma_type=int(action[8]), slow_ma_type=int(action[9]), signal_ma_type=int(action[10]))
        return self.exec_env()


class MACDOptimizeSavingFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _MACDExecuteFuturesEnv(*args, **kwargs)
        # To keep compatibility with gym env standards
        obs_lower_bounds = array([-inf for _ in range(1)])
        obs_upper_bounds = array([inf for _ in range(1)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        action_lower = [0.01, 0.000, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 2, 2, 2, 0, 0, 0, 1]
        action_upper = [1.00, 1.000, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 1_000, 1_000, 1_000, 36, 36, 25, 125]
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, *args, **kwargs):
        return self.exec_env.reset(*args, **kwargs)

    def step(self, action):
        self.reset(position_ratio=action[0], save_ratio=action[1], stop_loss=action[2], take_profit=action[3],
                   long_enter_at=action[4], long_close_at=action[5],
                   short_enter_at=action[6], short_close_at=action[7],
                   fast_period=int(action[8]), slow_period=int(action[9]), signal_period=int(action[10]),
                   fast_ma_type=int(action[11]), slow_ma_type=int(action[12]), signal_ma_type=int(action[13]),
                   leverage=int(action[14]))
        return self.exec_env()
