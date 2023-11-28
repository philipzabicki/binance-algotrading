import matplotlib.pyplot as plt
import get_data
from datetime import datetime
from dateutil.parser import parse

import TA_tools
from utilityx import minutes_since, get_slips_stats
from enviroments.backtest import BacktestEnvSpot

if __name__=='__main__':
    #df = TA_tools.get_combined_intervals_df(ticker='BTCUSDT', interval_list=['5m'], type='um', data='klines', template='basic')
    df = get_data.by_DataClient(ticker='BTCTUSD', interval='1m', futures=False)
    df = df.iloc[-minutes_since('22-03-2023'):,:].copy()
    df.to_csv('testo.csv', index=False)
    df = TA_tools.signal_only_features(df)
    strat_env = BacktestEnvSpot(df=df.drop(columns='Opened').to_numpy(), dates_df=df['Opened'].to_numpy(), excluded_left=0, init_balance=1_000, fee=0.0, slippage=get_slips_stats(), StopLoss=0.01, postition_ratio=1.0, Render_range=50, visualize=False, write_to_csv=False)
    #strat_env = BacktestEnvSpot(df=df[-1_109_336:,:], excluded_left=0, init_balance=1_000, fee=0.0, slippage=slippages, StopLoss=0.0001, postition_ratio=1.0, leverage=1, lookback_window_size=1, Render_range=30, visualize=False, dates_df=dates_df[-1_109_336:], write_to_csv=True)

    strat_env.reset()
    done = False
    action = 0
    while not done:
        obs,reward,done,info = strat_env.step(action)
        if strat_env.qty == 0:
            if obs[-1]>=5: action = 1
            elif obs[-1]<=-5: action = 2
            else: action = 0
        elif strat_env.qty<0:
            if obs[-1]>=5: action = 1
            else: action = 0
        elif strat_env.qty>0:
            if obs[-1]<=-5: action = 2
            else: action = 0
        if strat_env.visualize: strat_env.render()
    #df[['Close', 'SUM']].tail(250).plot.line()

    '''# Plot the first column 'A' and use its scale on the left y-axis
    ax1 = df['Close'].tail(250).plot(color='blue', label='A')
    ax1.set_ylabel('Scale for A', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    # Use twinx to create a second y-axis for the column 'B'
    ax2 = ax1.twinx()
    df['SUM'].tail(250).plot(ax=ax2, color='red', label='B')
    ax2.set_ylabel('Scale for B', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.show()'''

    #df.to_csv('test.csv', index=False)
    print(df.tail(25))