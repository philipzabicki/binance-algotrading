from multiprocessing import Pool, cpu_count
from statistics import mean, stdev

from matplotlib import pyplot as plt
from numpy import inf
from talib import AD

from enviroments.chaikinosc_env import ChaikinOscillatorStratFuturesEnv
from utils.get_data import by_BinanceVision
from utils.ta_tools import get_1D_MA, ChaikinOscillator_signal
from utils.utility import get_slippage_stats

CPU_CORES = cpu_count()
N_TEST = 10_000
N_STEPS = 2_880
TICKER, ITV, MARKET_TYPE, DATA_TYPE, START_DATE = 'BTCUSDT', '15m', 'um', 'klines', '2020-01-01'
ENV = ChaikinOscillatorStratFuturesEnv
ACTION = [0.9828319340319241, 0.013824254316664305, 5, 51, 372, 13, 3]


def sig_map(value):
    """Maps signals into values actually used by macd strategy env"""
    if 0 <= value < 0.5:
        return 0.5
    elif 0.5 <= value < 0.75:
        return 0.75
    else:
        return 1.0


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
    # print(f'pool{pool_nb}')
    # print(f'results{results}')
    # print(f'gains{gains}')
    return results, gains


if __name__ == "__main__":
    # df = pd.read_csv("C:/github/binance-algotrading/.other/lotos.csv")
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
    print(df)

    adl = AD(df['High'], df['Low'], df['Close'], df['Volume']).to_numpy()
    # print(adl[:10])
    # sleep(100)
    fast_adl, slow_adl = get_1D_MA(adl, ACTION[-2], ACTION[-4]), get_1D_MA(adl, ACTION[-1], ACTION[-3])
    chosc = fast_adl - slow_adl
    signals = ChaikinOscillator_signal(chosc)
    df['ADL'] = adl
    df['fast_ADL'] = fast_adl
    df['slow_ADL'] = slow_adl
    df['ChaikinOscillator'] = chosc
    df['signals'] = signals
    # df = df.tail(250)

    fig = plt.figure(figsize=(21, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1])
    axs = gs.subplots(sharex=False)
    axs[0].plot(df['fast_ADL'], label='fast_ADL')
    axs[0].plot(df['slow_ADL'], label='slow_ADL')
    axs[0].plot(df['ADL'], label='ADL')
    axs[0].legend(loc='upper left')
    axs[1].plot(df['ChaikinOscillator'], label='ChaikinOscillator')
    axs[1].legend(loc='upper left')
    axs[2].plot(df['signals'], label='Trade signals')
    axs[2].axhline(y=sig_map(ACTION[1]), label='Buy threshold', color='green', linestyle='--')
    axs[2].axhline(y=-sig_map(ACTION[2]), label='Sell threshold', color='red', linestyle='--')
    axs[2].legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    # plt.subplot(2, 1, 1)
    # plt.plot(macd[-1_000:], label='MACD')
    # plt.plot(signal[-1_000:], label='Signal')
    # plt.legend()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(signals[-1_000:], label='Signals')
    # plt.axhline(y=sig_map(action[1]), color='green', linestyle='--')
    # plt.axhline(y=-sig_map(action[2]), color='red', linestyle='--')
    # plt.legend()
    #
    # plt.show()

    with Pool(processes=CPU_CORES) as pool:
        # Each process will call 'run_indefinitely_process'
        # The list(range(num_processes)) is just to provide a different argument to each process (even though it's not used in the function)
        results = pool.starmap(parallel_test, [(i, df.iloc[:, 0:5], df_mark, dates_df) for i in range(CPU_CORES)])
    joined_res = []
    joined_gains = []
    for el in results:
        joined_res.extend(el[0])
        joined_gains.extend(el[1])
    # print(joined_res)
    # print(joined_gains)
    # print(len(set(joined_res)) == len(joined_res))
    # print(len(set(joined_res)))
    # print(len(joined_res))
    profitable = sum(i > 0 for i in joined_res)
    print(f'From {len(joined_res)} tests, profitable: {profitable} ({profitable / len(joined_res) * 100}%)')
    print(f'gain(avg/stdev): ${mean(joined_gains):_.2f}/${stdev(joined_gains):_.2f}')
    print(f'gain(min/max): ${min(joined_gains):_.2f}/${max(joined_gains):_.2f}')
