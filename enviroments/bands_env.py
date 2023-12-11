from numpy import array, float64, inf
from gym import spaces, Env
from enviroments import SignalExecuteSpotEnv
from utils.TA_tools import get_MA_band_signal
from utils.get_data import by_BinanceVision
from utils.utility import get_slippage_stats

class BandsExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=1, atr_period=1, atr_multi=1.0, **kwargs):
        super().reset(*args, **kwargs)
        self.stop_loss = stop_loss
        self.enter_threshold = enter_at
        self.close_threshold = close_at
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        self.df[self.start_step:self.end_step, -1] = get_MA_band_signal(self.df[self.start_step:self.end_step, :5],
                                                                        self.ma_type, self.ma_period,
                                                                        self.atr_period, self.atr_multi)
        self.obs = iter(self.df[self.start_step:self.end_step, :])
        return next(self.obs)

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' ma_type={self.ma_type}, ma_period={self.ma_period}')
            print(f' atr_period={self.atr_period}, atr_multi={self.atr_multi}')


class BandsStratSpotEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = BandsExecuteSpotEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.0001, 0.001, 0.001, 0, 2, 1, 0.001]
        action_upper = [0.0150, 1.000, 1.000, 35, 1_000, 1_000, 15.0]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, stop_loss=None, enter_at=1.0, close_at=1.0,
              ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.0):
        return self.exec_env.reset(stop_loss=stop_loss, enter_at=enter_at, close_at=close_at,
                                   ma_type=ma_type, ma_period=ma_period,
                                   atr_period=atr_period, atr_multi=atr_multi)

    def step(self, action):
        _reset = self.reset(stop_loss=action[0], enter_at=action[1], close_at=action[2],
                            ma_type=int(action[3]), ma_period=int(action[4]),
                            atr_period=int(action[5]), atr_multi=action[6])
        return self.exec_env(_reset)


# from matplotlib import pyplot as plt
if __name__ == "__main__":
    action = [0.010263476023283948, 0.19505459912279557, 0.733417664265096, 10, 224, 917, 6.808980697510462]
    dates_df, df = by_BinanceVision(ticker='BTCFDUSD',
                                    interval='1m',
                                    market_type='spot',
                                    data_type='klines',
                                    start_date='2023-09-11 00:00:00',
                                    split=True,
                                    delay=129_600)
    # macd, signal = custom_MACD(df.to_numpy(), action[3], action[4], action[5], action[6], action[7], action[8])
    # print(f'macd {macd}')
    # print(f'signal {signal}')
    # plt.plot(macd[-10_000:])
    # plt.plot(signal[-10_000:])
    # plt.show()

    env = BandsStratSpotEnv(df=df, dates_df=dates_df, init_balance=300, no_action_finish=inf,
                            fee=0.0, coin_step=0.00001,
                            slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
                            verbose=True, visualize=False)
    env.step(action)