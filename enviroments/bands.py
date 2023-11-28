# from os import environ
# environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

from numpy import array, float64, inf
from gym import spaces, Env
from enviroments.backtest import SpotBacktest, FuturesBacktest
from TA_tools import get_MA_signal


# from matplotlib import pyplot as plt
# import cProfile

class OneRunEnv(SpotBacktest):
    def __init__(self, df, dates_df=None,
                 exclude_cols_left=0,
                 max_steps=0,
                 stop_loss=0.0,
                 enter_at=1.000, close_at=-1.000,
                 ma_type=0, ma_period=2,
                 atr_period=2, atr_multi=1.000,
                 init_balance=100_000,
                 position_ratio=1.00,
                 fee=0.0002, coin_step=0.00001,
                 slippage=None,
                 render_range=120,
                 visualize=False):
        # self.profiler = cProfile.Profile()
        # self.profiler.enable()
        super().__init__(df=df,
                         dates_df=dates_df,
                         exclude_cols_left=exclude_cols_left,
                         init_balance=init_balance,
                         position_ratio=position_ratio,
                         stop_loss=stop_loss,
                         fee=fee,
                         coin_step=coin_step,
                         slippage=slippage,
                         max_steps=max_steps,
                         render_range=render_range,
                         visualize=visualize)
        self.enter_threshold = enter_at
        self.close_threshold = close_at
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi

    def reset(self, position_ratio=1.00, stop_loss=0.01, enter_at=1.000, close_at=-1.000, ma_type=0, ma_period=1,
              atr_period=1, atr_multi=1.000):
        self.stop_loss = stop_loss
        self.enter_threshold = enter_at
        self.close_threshold = close_at
        self.position_ratio = position_ratio
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        # print(f'OneRunEnv.reset {self.postition_ratio} {self.stop_loss} {self.enter_threshold} {self.close_threshold} {self.ma_type} {self.ma_period} {self.atr_period} {self.atr_multi}')
        super().reset()
        self.df[self.start_step:self.end_step, -1] = get_MA_signal(self.df[self.start_step:self.end_step, :],
                                                                   self.ma_type, self.ma_period, self.atr_period,
                                                                   self.atr_multi)
        self.obs = iter(self.df[self.start_step:self.end_step, :])
        return array([-1.0 for _ in range(7)], dtype="float32")

    def _finish_episode(self):
        # print('OneRunEnv._finish_episode()')
        super()._finish_episode()
        # obs =  array([self.episode_orders, self.cumulative_fees, self.sharpe_ratio, self.sortino_ratio, self.max_drawdown, self.PL_count_mean, self.PL_ratio])
        if self.output:
            print(f' postition_ratio={self.position_ratio}, stop_loss={self.stop_loss:.4f}', end='')
            print(f' enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}', end='')
            print(f' ma_type={self.ma_type}, ma_period={self.ma_period}', end='')
            print(f' atr_period={self.atr_period}, atr_multi={self.atr_multi:.3f}', end='')
            print(f' reward={self.reward:.2f} (exec_time={self.info["exec_time"]:.2f}s)')
        # self.profiler.disable()  # Zakończ profilowanie
        # self.profiler.print_stats(sort='cumtime')
        # return obs, reward, done, info

    def _execute(self):
        # The first action must be initiated, I used 0
        obs, reward, done, info = [None] * 4
        # The first action must be initiated, I used
        action = 0
        while not self.done:
            obs, reward, done, info, *_ = self.step(action)
            if self.visualize: self.render()
            # print(f'SIGNAL: {obs[-1]:.2f}')
            # obs[-1] asummes that indicator vlaues are stored in last column
            signal = obs[-1]
            action = 0
            if self.qty == 0:
                if signal >= self.enter_threshold:
                    action = 1
                elif signal <= -self.enter_threshold:
                    action = 2
            elif (self.qty < 0) and (signal >= -self.close_threshold):
                action = 1
            elif (self.qty > 0) and (signal <= self.close_threshold):
                action = 2
        # print(obs,reward,done,info)
        return obs, reward, done, info


class BandsStratEnv(Env):
    def __init__(self, df, dates_df=None, excluded_left=0, init_balance=100_000, position_ratio=1.0,
                 fee=0.0002, coin_step=0.00001, slippage=None,
                 max_steps=0, render_range=120, visualize=False, exclude_cols_left=None):
        self.exec_env = OneRunEnv(df, dates_df=dates_df, stop_loss=0.01, ma_type=0, ma_period=2, atr_period=2,
                                  atr_multi=1.000,
                                  exclude_cols_left=exclude_cols_left, init_balance=init_balance,
                                  position_ratio=position_ratio,
                                  fee=fee, coin_step=coin_step, slippage=slippage, max_steps=max_steps,
                                  render_range=render_range, visualize=visualize)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 0.001]
        action_upper = [0.0150, 1.000, 1.000, 35, 1_000, 1_000, 15.000]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, postition_ratio=1.0, stop_loss=0.01, enter_at=1.000, close_at=-1.000, ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.000):
        # print(f'BandsStratEnv.reset {postition_ratio} {stop_loss} {enter_at} {close_at} {ma_type} {ma_period} {atr_period} {atr_multi}')
        return self.exec_env.reset(postition_ratio, stop_loss, enter_at, close_at, ma_type, ma_period, atr_period,
                                   atr_multi)

    def step(self, action):
        # print(f'action: {action}')
        self.reset(postition_ratio=1.0, stop_loss=round(action[0], 4),
                   enter_at=round(action[1], 3), close_at=-round(action[2], 3),
                   ma_type=int(action[3]), ma_period=int(action[4]),
                   atr_period=int(action[5]), atr_multi=round(action[6], 3))
        return self.exec_env._execute()


class OneRunEnvFutures(FuturesBacktest):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.ma_type = kwargs['ma_type']
        self.ma_period = kwargs['ma_period']
        self.atr_period = kwargs['atr_period']
        self.atr_multi = kwargs['atr_multi']


class BandsStratEnvFutures(Env):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
