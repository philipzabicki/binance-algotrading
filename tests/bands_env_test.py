from multiprocessing import Pool, cpu_count

from matplotlib import pyplot as plt
from numpy import inf, mean, std

from enviroments.bands_env import BandsStratFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_MA_band_signal
from utils.utility import get_slippage_stats

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 2_880
TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '15m', 'um', 'klines', '2020-01-01'
ENV = BandsStratFuturesEnv
ACTION = [0.08057522386480123, 0.005935781231366253, 0.6091806568690151, 0.590162661946407, 9.346681371369177, 12, 14, 185, 63]


def parallel_test(pool_nb, df, df_mark=None, dates_df=None):
    env = ENV(df=df,
              df_mark=df_mark,
              dates_df=dates_df,
              max_steps=N_STEPS,
              init_balance=350,
              no_action_finish=inf,
              fee=0.0005,
              coin_step=0.001,
              # slipp_std=0,
              slippage=get_slippage_stats('spot', 'BTCFDUSD', '1m', 'market'),
              verbose=False, visualize=False, write_to_file=True)
    results, gains = [], []
    for _ in range(N_TEST // CPU_CORES):
        _, reward, _, _, _ = env.step(ACTION)
        results.append(reward)
        if reward > 0:
            gains.append(env.exec_env.balance - env.exec_env.init_balance)
    return results, gains


if __name__ == "__main__":
    dates_df, df = by_BinanceVision(ticker=TICKER,
                                    interval=ITV,
                                    market_type=MARKET_TYPE,
                                    data_type=DATA_TYPE,
                                    start_date=START_DATE,
                                    split=True,
                                    delay=259_200)
    _, df_mark = by_BinanceVision(ticker=TICKER,
                                  interval=ITV,
                                  market_type=MARKET_TYPE,
                                  data_type='markPriceKlines',
                                  start_date=START_DATE,
                                  split=True,
                                  delay=259_200)
    signals = get_MA_band_signal(df.to_numpy(), ACTION[-3], ACTION[-2], ACTION[-4], ACTION[-5])
    print(f'signal {signals}')
    plt.plot(signals)
    plt.axhline(y=ACTION[2], color='green', linestyle='--')
    plt.axhline(y=-ACTION[3], color='red', linestyle='--')
    plt.show()

    with Pool(processes=CPU_CORES) as pool:
        results = pool.starmap(parallel_test, [(i, df.iloc[:, 0:5], df_mark, dates_df) for i in range(CPU_CORES)])
    joined_res = []
    joined_gains = []
    for el in results:
        joined_res.extend(el[0])
        joined_gains.extend(el[1])
    profitable = sum(i > 0 for i in joined_res)
    print(f'From {len(joined_res)} tests, profitable: {profitable} ({profitable / len(joined_res) * 100}%)')
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${std(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
