from enviroments.dca_env import DCAEnv
from utils.get_data import by_BinanceVision

if __name__ == "__main__":
    results = []
    for day in range(1, 31):
        print(f'Day of month: {day}')
        for hour in range(24):
            print(f'hour of day: {hour}', end=' ')
            # [368:-218]
            _, df = by_BinanceVision(ticker='XRPUSDT',
                                     interval='1m',
                                     market_type='um',
                                     data_type='klines',
                                     start_date='2021-01-01',
                                     split=True,
                                     delay=0)
            strat_env = DCAEnv(df=df, excluded_left=0, init_balance=29, fee=0.0, slippage=0.0001,
                               position_ratio=0.00001, leverage=1, lookback_window_size=1, Render_range=50,
                               visualize=False)
            obs, _, done, _, info = strat_env.reset()
            # print(obs)
            _counter = 0
            while not strat_env.done:
                # obs_current_date = dt.fromtimestamp(obs[0])
                if obs[0].hour == hour and obs[0].day == day:
                    action = 1
                    _counter += 1
                    # print(f'({obs[0]} should buy {_counter})')
                    # time.sleep(0.1)
                else:
                    action = 0
                obs, _, done, info = strat_env.step(action)
            results.append([day, hour, info['asset_balance'], info['purchase_count']])
            # time.sleep(1)
            # print(results)
    results = [el for el in results if el[0] < 29]
    print(sorted(results, key=lambda item: item[2]))
    # print(obs)
