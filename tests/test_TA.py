import TA_tools
from utility import seconds_since
from get_data import by_BinanceVision
import talib
from finta import TA as finTA
from tindicators import ti
from numpy import array
import pandas as pd
from matplotlib import pyplot as plt
from random import randint

if __name__=='__main__':
    df = by_BinanceVision(ticker='BTCFDUSD',
                          interval='1s',
                          type='spot',
                          data='klines',
                          delay=604_800).tail(seconds_since('09-11-2023'))
    df.drop(columns=['Opened'], inplace=True)
    print(df)
    lower_idx = randint(0, len(df))
    # lower_idx = 500

    rsi = talib.RSI(df['Close'], timeperiod=14)
    rsi_sig = TA_tools.RSI_like_signal(rsi.to_numpy(), 14)
    ult = talib.ULTOSC(df['High'], df['Low'], df['Close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ult_sig = TA_tools.RSI_like_signal(ult.to_numpy(), 28)
    mfi = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
    mfi_sig = TA_tools.RSI_like_signal(mfi.to_numpy(), 14)

    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    macd_cross_sig = TA_tools.MACD_cross_signal(macd.to_numpy(), macdsignal.to_numpy())
    macd_zerocross_sig = TA_tools.MACD_zerocross_signal(macd.to_numpy(), macdsignal.to_numpy())
    macd_reversal_sig = TA_tools.MACDhist_reversal_signal(macdhist.to_numpy())

    up, mid, low = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=talib.MA_Type.EMA)
    BB_sig = TA_tools.BB_signal(df['Close'].to_numpy(), up, mid, low)

    mDI = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    pDI = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    adx = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    adx_signal = TA_tools.ADX_signal(adx.to_numpy(), mDI.to_numpy(), pDI.to_numpy())
    adxtrend_signal = TA_tools.ADX_trend_signal(adx.to_numpy(), mDI.to_numpy(), pDI.to_numpy())

    frama = ti.frama(df['High'], df['Low'], 2*int(19//2), 2*int(19//2)//2)
    plt.plot(frama[-1_000:])
    plt.show()
    print(frama)

    sum = array(adx_signal)+array(adxtrend_signal)+array(rsi_sig)+array(ult_sig)+array(mfi_sig)+array(macd_cross_sig)+array(macd_zerocross_sig)+array(macd_reversal_sig)+array(BB_sig)
    SMAsum = talib.EMA(sum,14)

    fig, axs = plt.subplots(3)
    axs[0].plot(df['Close'][-lower_idx-200:-lower_idx])
    axs[1].plot(df['Volume'][-lower_idx-200:-lower_idx])
    axs[2].plot(frama[-lower_idx-200:-lower_idx])
    #axs[0].plot(up[-lower_idx-200:-lower_idx])
    #axs[0].plot(low[-lower_idx-200:-lower_idx])
    #axs[1].plot(macd[-lower_idx-200:-lower_idx])
    #axs[1].plot(macdsignal[-lower_idx-200:-lower_idx])
    #axs[1].plot(BB_sig[-lower_idx-200:-lower_idx])
    '''axs[1].plot(adx[-lower_idx-200:-lower_idx])
    axs[1].plot(mDI[-lower_idx-200:-lower_idx])
    axs[1].plot(pDI[-lower_idx-200:-lower_idx])'''
    #axs[3].plot(macd_zerocross_sig[-lower_idx-200:-lower_idx])
    #axs[4].plot(macd_reversal_sig[-lower_idx-200:-lower_idx])
    #axs[2].plot(sum[-lower_idx-200:-lower_idx])
    #axs[3].plot(SMAsum[-lower_idx-200:-lower_idx])
    
    plt.show()