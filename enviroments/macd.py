from numpy import array, float64, inf, zeros, hstack
from gym import spaces, Env
from enviroments.single_signal import SignalExecuteEnv
from talib import MACDEXT
from TA_tools import custom_MACD, MACD_cross_signal
from utility import minutes_since, seconds_since, get_market_slips_stats
from get_data import by_BinanceVision

class MACDExecuteEnv(SignalExecuteEnv):
    def reset(self, *args, fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, **kwargs):
        super().reset(*args, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        macd, macd_signal = custom_MACD(self.df[self.start_step:self.end_step, :5],
                                           fast_ma_type=fast_ma_type, fast_period=fast_period,
                                           slow_ma_type=slow_ma_type, slow_period=slow_period,
                                           signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.df[self.start_step:self.end_step, -1] = MACD_cross_signal(macd, macd_signal)
        # print(f'1: {count_nonzero(self.df[self.start_step:self.end_step, -1]>=1)}')
        # print(f'-1: {count_nonzero(self.df[self.start_step:self.end_step, -1]<=-1)}')
        self.obs = iter(self.df[self.start_step:self.end_step, :])
        return next(self.obs)
        # return self.df[self.current_step, :]

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' fast_period={self.fast_period}, slow_period={self.slow_period}, signal_period={self.signal_period}')
            print(f' fast_MA_type={self.fast_ma_type}, slow_MA_type={self.slow_ma_type}, signal_MA_type={self.signal_ma_type}')

class MACDStratEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = MACDExecuteEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [2, 2, 2, 0, 0, 0]
        action_upper = [10_000, 10_000, 10_000, 34, 34, 25]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=1, slow_ma_type=1, signal_ma_type=1):
        # print(f'BandsStratEnv.reset {postition_ratio} {stop_loss} {enter_at} {close_at} {ma_type} {ma_period} {atr_period} {atr_multi}')
        return self.exec_env.reset(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period,
                                   fast_ma_type=fast_ma_type, slow_ma_type=slow_ma_type, signal_ma_type=signal_ma_type)

    def step(self, action):
        _reset = self.reset(fast_period=int(action[0]), slow_period=int(action[1]), signal_period=int(action[2]),
                   fast_ma_type=int(action[3]), slow_ma_type=int(action[4]), signal_ma_type=int(action[5]))
        return self.exec_env(_reset)


if __name__ == "__main__":
    action = [5, 6, 3, 23, 13, 12]
    df = by_BinanceVision(ticker='BTCFDUSD', interval='1m', type='spot', data='klines', delay=129_600)
    dates_df = df['Opened'].to_numpy()[-minutes_since('11-09-2023'):]
    df = df.drop(columns='Opened').to_numpy()[-minutes_since('11-09-2023'):, :]
    df = hstack((df, zeros((df.shape[0], 1))))
    env = MACDStratEnv(df=df, dates_df=dates_df, init_balance=300, no_action_finish=inf,
                       fee=0.0, coin_step=0.00001,
                       # slippage=get_market_slips_stats(),
                       verbose=True, visualize=False)
    env.step(action)