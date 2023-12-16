from gym import spaces, Env
from numpy import array, float64, inf
from talib import RSI

from utils.ta_tools import custom_MACD, MACD_cross_signal, RSI_like_signal
from .base import SignalExecuteSpotEnv


class MACDRSIExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None,
              enter_at1=1.0, close_at1=1.0,
              enter_at2=1.0, close_at2=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0,
              rsi_period, **kwargs):
        super().reset(*args, **kwargs)
        self.stop_loss = stop_loss
        self.enter_threshold1 = enter_at1
        self.enter_threshold2 = enter_at2
        self.close_threshold1 = close_at1
        self.close_threshold2 = close_at2
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        self.rsi_period = rsi_period
        macd, macd_signal = custom_MACD(self.df[self.start_step:self.end_step, :5],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        rsi = RSI(self.df[self.start_step:self.end_step, 3], self.rsi_period)
        self.signals1 = MACD_cross_signal(macd, macd_signal)
        # print(rsi[-5:])
        # print(self.signals1[-5:])
        self.signals2 = RSI_like_signal(rsi, self.rsi_period)
        self.obs = iter(self.df[self.start_step:self.end_step, :])
        return next(self.obs)

    def __call__(self, *args, **kwargs):
        while not self.done:
            action = 0
            if self.qty == 0:
                if (self.signals1[self.current_step] >= self.enter_threshold1) and (
                        self.signals2[self.current_step] >= self.enter_threshold2):
                    action = 1
                elif (self.signals1[self.current_step] <= -self.enter_threshold1) and (
                        self.signals2[self.current_step] <= -self.enter_threshold2):
                    action = 2
            elif (self.qty < 0) and (self.signals1[self.current_step] >= self.close_threshold1) and (
                    self.signals2[self.current_step] >= self.close_threshold2):
                action = 1
            elif (self.qty > 0) and (self.signals1[self.current_step] <= -self.close_threshold1) and (
                    self.signals2[self.current_step] <= -self.close_threshold2):
                action = 2
            obs, _, _, _, _ = self.step(action)
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render()
                self.current_step += 1
        return obs, self.reward, self.done, False, self.info

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' enter_at1={self.enter_threshold1:.3f}, close_at1={self.close_threshold1:.3f}')
            print(f' enter_at2={self.enter_threshold2:.3f}, close_at2={self.close_threshold2:.3f}')
            print(
                f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(
                f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')
            print(f' rsi_period={self.rsi_period}')


class MACDRSIStratSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = MACDRSIExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 0.001, 0.001, 2, 2, 2, 0, 0, 0, 2]
        action_upper = [0.0500, 1.000, 1.000, 1.000, 1.000, 10_000, 10_000, 10_000, 37, 37, 26, 10_000]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, enter_at1=1.0, close_at1=1.0,
              enter_at2=1.0, close_at2=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=1, slow_ma_type=1, signal_ma_type=1,
              rsi_period=14):
        return self.exec_env.reset(stop_loss=stop_loss, enter_at1=enter_at1, close_at1=close_at1,
                                   enter_at2=enter_at2, close_at2=close_at2,
                                   fast_period=fast_period, slow_period=slow_period, signal_period=signal_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type, signal_ma_type=signal_ma_type,
                                   rsi_period=rsi_period)

    def step(self, action):
        # print(action)
        self.reset(stop_loss=action[0], enter_at1=action[1], close_at1=action[2], enter_at2=action[3],
                   close_at2=action[4],
                   fast_period=int(action[5]), slow_period=int(action[6]), signal_period=int(action[7]),
                   fast_ma_type=int(action[8]), slow_ma_type=int(action[9]), signal_ma_type=int(action[10]),
                   rsi_period=int(action[11]))
        return self.exec_env()
