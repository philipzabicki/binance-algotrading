import pandas as pd
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from time import time
from statistics import mean, stdev
from math import sqrt, copysign
from finta import TA as finTA
import talib
import ta 
import get_data
from utility import linear_reg_slope
from numba import jit
#import MAs
#import datetime as dt
#from numba import njit, jit
#from scipy.ndimage import convolve1d as conv
#from sklearn import preprocessing

def feature_timeit(feature_func):
  def wrapper(*args, **kwargs):
    start_t = time()
    print(f'\r adding {feature_func.__name__} feature...', end='')
    ret = feature_func(*args, **kwargs)
    print(f' ({(time()-start_t):.3f}s)')
    return ret
  return wrapper

@feature_timeit
def scaleColumns(df, scaler):
    for col in df.columns:
      if col not in ['Open time', 'Opened', 'Close time', 'Open', 'High', 'Low', 'Close']:
        #print(col)
        #caler.fit(df[[col]])
        df[col] = scaler.fit_transform(df[[col]])
    return df

def linear_slope_indicator(self, values: list):
      _5 = len(values)//20
      percentile25 = linear_reg_slope(values[-_5*5:])
      #percentile50 = linear_reg_slope(values[-_5*10:])
      #percentile75 = linear_reg_slope(values[-_5*15:])
      #percentile95 = linear_reg_slope(values[-_5*19:])
      slope_avg = percentile25
      #return slope_avg
      return copysign(abs(slope_avg)**(1/4), slope_avg)

######################################################################################
############################# SIGNAL GENERATORS ######################################
######################################################################################
@feature_timeit
def ULT_RSI_signal(column, timeperiod):
  return [None] * timeperiod + [  -1 if cur >= 65.0 else
                                  1 if cur <= 35.0 else
                                  -2 if prev>65.0 and cur <= 65.0 else
                                  2 if prev<35.0 and cur >= 35.0 else
                                  0
                                  for cur, prev in zip(column[timeperiod:], column[timeperiod-1:-1])  ]

@feature_timeit
def ADX_signal(adx_col, minus_DI, plus_DI):
  return [0] + [  1 if cur_pDI>cur_mDI and prev_pDI<prev_mDI and adx>25.0 else
                  .75 if cur_pDI>cur_mDI and prev_pDI<prev_mDI and adx>20.0 else
                  -1 if cur_pDI<cur_mDI and prev_pDI>prev_mDI and adx>25.0 else
                  -.75 if cur_pDI<cur_mDI and prev_pDI>prev_mDI and adx>20.0 else
                  .5 if cur_pDI>prev_pDI and cur_mDI<prev_mDI and cur_pDI<cur_mDI and adx>25.0 else
                  -.5 if cur_pDI<prev_pDI and cur_mDI>prev_mDI and cur_pDI>cur_mDI and adx>25.0 else
                  .25 if cur_pDI>prev_pDI and cur_mDI<prev_mDI and cur_pDI<cur_mDI and adx>20.0 else
                  -.25 if cur_pDI<prev_pDI and cur_mDI>prev_mDI and cur_pDI>cur_mDI and adx>20.0 else
                  0
                  for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(plus_DI[1:], minus_DI[1:], adx_col[1:], plus_DI[:-1], minus_DI[:-1]) ]

@feature_timeit
def ADX_trend(column):
  return [ 1 if val>25.0 else -1 if val<20.0 else 0 for val in column ]

@feature_timeit
def MFI_signal(mfi_col):
  def _sig(val):
    if val>90: return 1
    elif val<10: return -1
    elif val>80: return .5
    elif val<20: return -.5
    else: return 0
  return [0] + [ _sig(mfi) for mfi in mfi_col[1:] ]

@feature_timeit
def MFI_divergence(mfi_col, close_col):
  return [0] + [ 1 if (prev_mfi<20 and cur_mfi>20) and (cur_close<prev_close) else 
                 -1 if (prev_mfi>80 and cur_mfi<80) and (cur_close>prev_close) else 
                 0 
                 for cur_mfi, cur_close, prev_mfi, prev_close in zip(mfi_col[1:], close_col[1:], mfi_col[:-1], close_col[:-1]) ]

@feature_timeit
def MACD_cross(macd_col, signal_col):
  return [0] + [  1 if cur_macd>cur_sig and prev_macd<prev_sig else
                  -1 if cur_macd<cur_sig and prev_macd>prev_sig else
                  0
                  for cur_sig, cur_macd, prev_sig, prev_macd in zip(signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]) ]

@feature_timeit
def MACDhist_reversal(macdhist_col):
  return [None] * 3 + [ 1 if cur_macd>prev_macd and prev_macd<preprev_macd<prepreprev_macd else
                        -1 if cur_macd<prev_macd and prev_macd>preprev_macd>prepreprev_macd else
                        0
                        for cur_macd, prev_macd, preprev_macd, prepreprev_macd in zip(macdhist_col[3:], macdhist_col[2:-1], macdhist_col[1:-2], macdhist_col[:-3]) ]

@feature_timeit
def MACD_zerocross(macd_col, signal_col):
  return [None] + [ .5 if cur_macd>0 and prev_macd<0 else
                    1 if cur_sig>0 and prev_sig<0 else
                    -.5 if cur_macd<0 and prev_macd>0 else
                    -1 if cur_sig<0 and prev_sig>0 else
                    0
                    for cur_macd, cur_sig, prev_macd, prev_sig in zip(macd_col[1:], signal_col[1:], macd_col[:-1], signal_col[:-1]) ]

@feature_timeit
def BB_sig(mid, up_x1, low_x1, up_x2, low_x2, up_x3, low_x3, close):
  return [  -1 if close[i]>up_x3[i] else
            1 if close[i]<low_x3[i] else
            -.75 if up_x2[i]<close[i]<up_x3[i] else
            .75 if low_x3[i]<close[i]<low_x2[i] else
            -.5 if up_x1[i]<close[i]<up_x2[i] else
            .5 if low_x2[i]<close[i]<low_x1[i] else
            -.25 if mid[i]<close[i]<up_x1[i] else
            .25 if low_x1[i]<close[i]<mid[i] else
            0
            for i in range(len(close))  ]

@feature_timeit
def Price_levels(Open, Close, decimals=0, sort=False):
  def get_levels(open, close, decimals=0, sort=False):
    tckr_lvls={}
    for open, close, close_next in zip(open[:-1], close[:-1], close[1:]):
      if (close>open>close_next) or (close<open<close_next):
        lvl = round(close, decimals)
        if lvl in tckr_lvls:
          tckr_lvls[lvl] += 1
        else:
          tckr_lvls[lvl] = 1
    if sort: tckr_lvls = { k:v for k,v in sorted(tckr_lvls.items(), key=lambda item: item[1], reverse=True) }
    return tckr_lvls
  lvls = get_levels(Open, Close, decimals=decimals, sort=sort)
  return [ lvls[round(c, decimals)] if round(c, decimals) in lvls.keys() else 0 for c in Close ]

@feature_timeit
def Move_probablity(Open, Close):
  def get_avg_changes(open, close):
    gain = [ (close/open-1)*100 for open,close in zip(open,close) if close>open ]
    loss = [ (open/close-1)*100 for open,close in zip(open,close) if close<open ]
    return mean(gain),stdev(gain),mean(loss),stdev(loss)
  avg_gain,gain_stdev,avg_loss,loss_stdev = get_avg_changes(Open, Close)
  return [ (((close/open-1)*100)-avg_gain)/gain_stdev if close>open else (((open/close-1)*100)-avg_loss)/loss_stdev for open,close in zip(Open,Close) ]

@feature_timeit
def Volume_probablity(Volume):
  return (Volume-np.mean(Volume))/np.std(Volume)

@feature_timeit
def Hourly_seasonality_from_other(ticker, Opened_hour, type='um', data='klines'):
  def get_hour_changes(ticker='BTCUSDT', type='um', data='klines'):
    tckr_df = get_data.by_BinanceVision(ticker, '1h', type=type, data=data)
    hour = tckr_df['Opened'].dt.hour.to_numpy()
    CO_chng = ((tckr_df['Close'].to_numpy()/tckr_df['Open'].to_numpy())-1)*100
    return { i:np.mean([chng for chng, h in zip(CO_chng, hour) if h==i]) for i in range(0,24) }
  h_dict = get_hour_changes(ticker=ticker, type=type, data=data)
  return [h_dict[h] for h in Opened_hour]

@feature_timeit
def Hourly_seasonality(df):
  hour = df['Opened'].dt.hour
  CO_chng = ((df['Close']/df['Open'])-1)*100
  h_dict = { i:np.mean([chng for chng, h in zip(CO_chng, hour) if h==i]) for i in range(0,24) }
  return [ h_dict[h] for h in hour ]

@feature_timeit
def Daily_seasonality_from_other(ticker, Opened_weekday, type='um', data='klines'):
  def get_weekday_changes(ticker='BTCUSDT', type='um', data='klines'):
    tckr_df = get_data.by_BinanceVision(ticker, '1d', type=type, data=data)
    weekday = tckr_df['Opened'].dt.dayofweek.to_numpy()
    CO_chng=((tckr_df['Close'].to_numpy()/tckr_df['Open'].to_numpy())-1)*100
    return { i:np.mean([chng for chng, day in zip(CO_chng, weekday) if day==i]) for i in range(0,7) }
  wd_dict = get_weekday_changes(ticker=ticker, type=type, data=data)
  return [ wd_dict[w] for w in Opened_weekday ]

@feature_timeit
def Daily_seasonality(df):
  weekday = df['Opened'].dt.dayofweek
  CO_chng=((df['Close']/df['Open'])-1)*100
  wd_dict = { i:np.mean([chng for chng, day in zip(CO_chng, weekday) if day==i]) for i in range(0,7) }
  return [ wd_dict[w] for w in weekday ]
######################################################################################
######################################################################################
######################################################################################

######################################################################################
############################# Nontypical MA functions ################################
######################################################################################
#@feature_timeit
def HullMA(close, timeperiod):
  return talib.WMA( (talib.WMA(close, timeperiod//2 )*2)-(talib.WMA(close, timeperiod)), int(np.sqrt(timeperiod)) )

#@feature_timeit
def RMA(close, timeperiod):
  alpha = 1.0 / timeperiod
  rma = [0.0] * len(close)
  # Calculating the SMA for the first 'length' values
  sma = sum(close[:timeperiod]) / timeperiod
  rma[timeperiod - 1] = sma
    # Calculating the rma for the rest of the values
  for i in range(timeperiod, len(close)):
    rma[i] = alpha * close[i] + (1 - alpha) * rma[i - 1]
  return rma

#@feature_timeit
def VWMA(close, volume, timeperiod):
    cum_sum = 0
    cum_vol = 0
    vwmas = []
    cv_list = [close[i] * volume[i] for i in range(len(close))]  # precompute close[i] * volume[i]
    i = 0
    while i<len(close):
        cum_sum += cv_list[i]
        cum_vol += volume[i]
        if i >= timeperiod:
            cum_sum -= cv_list[i - timeperiod]
            cum_vol -= volume[i - timeperiod]
        vwma = cum_sum / cum_vol
        vwmas.append(vwma)
        i += 1
    return vwmas

#@feature_timeit
def ALMA(close, timeperiod, offset=0.85, sigma=6):
    m = offset * (timeperiod - 1)
    s = timeperiod / sigma
    wtd = np.array([np.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(timeperiod)])
    wtd /= sum(wtd)
    alma = np.convolve(close, wtd, mode='valid')
    return np.insert(alma, 0, [np.nan]*(timeperiod-1))

#@feature_timeit
def HammingMA(close, timeperiod):
    w = np.hamming(timeperiod)
    hma = np.convolve(close, w, mode='valid') / w.sum()
    return np.insert(hma, 0, [np.nan]*(timeperiod-1))

'''#@feature_timeit
def LSMA(close, timeperiod):
    lsma = []
    for i in range(timeperiod - 1, len(close)):
        x = np.arange(0, timeperiod)  # zawsze generuj x jako zakres od 0 do timeperiod-1
        y = close[i - timeperiod + 1:i + 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        lsma.append(m * (timeperiod-1) + c)  # wykorzystaj timeperiod-1 zamiast i do obliczenia prognozy
    return np.array([np.nan for _ in range(timeperiod-1)]+lsma)'''

@jit(nopython=True)
def LSMA(close, timeperiod):
    n = len(close)
    lsma = np.empty(n)
    lsma[:timeperiod-1] = np.nan
    # Przygotuj x raz i używaj w pętli
    x = np.arange(0, timeperiod)
    A = np.empty((timeperiod, 2))
    A[:, 0] = x
    A[:, 1] = 1
    AT = A.T.copy()  # This creates a contiguous array
    ATA_inv = np.linalg.inv(np.dot(AT, A))
    for i in range(timeperiod - 1, n):
        y = close[i - timeperiod + 1:i + 1].copy()
        m, c = np.dot(ATA_inv, np.dot(AT, y))
        lsma[i] = m * (timeperiod-1) + c
    return lsma

'''#@feature_timeit
def NadarayWatsonMA(close, timeperiod, kernel=0):
    # Initialize the Nadaray-Watson moving average array with NaNs for the start period
    nwma = np.empty_like(close)
    nwma[:timeperiod-1] = np.nan
    # Define the Epanechnikov kernel function
    def gaussian_kernel(x):
      return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    def epanechnikov_kernel(x):
      return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)
    def rectangular_kernel(x):
      return np.where(np.abs(x) <= 1, 0.5, 0)
    def triangular_kernel(x):
      return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
    def biweight_kernel(x):
      return np.where(np.abs(x) <= 1, (15/16) * (1 - x**2)**2, 0)
    def cosine_kernel(x):
      return np.where(np.abs(x) <= 1, np.pi/4 * np.cos(np.pi/2 * x), 0)
    kernel_func = {0:gaussian_kernel,1:epanechnikov_kernel,2:rectangular_kernel,3:triangular_kernel,4:biweight_kernel,5:cosine_kernel}
    for i in range(timeperiod-1, len(close)):
        window_prices = close[i-timeperiod+1:i+1]
        distances = np.abs(np.arange(timeperiod) - (timeperiod-1))
        weights = kernel_func[kernel](distances / timeperiod)
        nwma[i] = (weights @ window_prices) / weights.sum()
    return nwma'''

@jit(nopython=True)
def gaussian_kernel(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
@jit(nopython=True)
def epanechnikov_kernel(x):
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)
@jit(nopython=True)
def rectangular_kernel(x):
    return np.where(np.abs(x) <= 1, 0.5, 0)
@jit(nopython=True)
def triangular_kernel(x):
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
@jit(nopython=True)
def biweight_kernel(x):
    return np.where(np.abs(x) <= 1, (15/16) * (1 - x**2)**2, 0)
@jit(nopython=True)
def cosine_kernel(x):
    return np.where(np.abs(x) <= 1, np.pi/4 * np.cos(np.pi/2 * x), 0)
@jit(nopython=True)
def NadarayWatsonMA(close, timeperiod, kernel=0):
    nwma = np.empty_like(close)
    nwma[:timeperiod-1] = np.nan
    for i in range(timeperiod-1, len(close)):
        window_prices = np.ascontiguousarray(close[i-timeperiod+1:i+1])
        distances = np.abs(np.arange(timeperiod) - (timeperiod-1))
        # Explicitly branching for kernels
        if kernel == 0:
            weights = gaussian_kernel(distances / timeperiod)
        elif kernel == 1:
            weights = epanechnikov_kernel(distances / timeperiod)
        elif kernel == 2:
            weights = rectangular_kernel(distances / timeperiod)
        elif kernel == 3:
            weights = triangular_kernel(distances / timeperiod)
        elif kernel == 4:
            weights = biweight_kernel(distances / timeperiod)
        elif kernel == 5:
            weights = cosine_kernel(distances / timeperiod)
        else:
            raise ValueError("Invalid kernel value")
        weights = np.ascontiguousarray(weights)
        nwma[i] = (weights @ window_prices) / weights.sum()
    return nwma

#@feature_timeit
def LWMA(close, period):
    weights = np.arange(1, period + 1)
    lwma = np.zeros_like(close)
    for i in range(period - 1, len(close)):
        lwma[i] = np.dot(weights, close[i - period + 1 : i + 1]) / weights.sum()
    return lwma

#@feature_timeit
def MGD(close, period):
    md = np.zeros_like(close)
    md[0] = close[0]
    for i in range(1, len(close)):
        md[i] = md[i-1] + (close[i] - md[i-1]) / (period * np.power((close[i] / md[i-1]), 4))
    return md

#@feature_timeit
def VIDYA(close, period):
    alpha = 2 / (period + 1)
    #k = talib.CMO(close, period)
    k = np.abs(talib.CMO(close, period))/100
    VIDYA = np.zeros_like(close)
    VIDYA[period-1] = close[period-1]
    for i in range(period, len(close)):
        VIDYA[i] = alpha * k[i] * close[i] + (1 - alpha * k[i]) * VIDYA[i-1]
    return VIDYA

#@feature_timeit
def GMA(close, period):
    """Compute Geometric Moving Average using logarithms for efficiency."""
    gma = np.zeros(len(close))
    log_close = np.log(close)
    for i in range(period-1, len(close)):
        gma[i] = np.exp(np.mean(log_close[i-period+1:i+1]))
    return gma

#@feature_timeit
def FBA(close, period):
    def fibonacci_sequence(n):
      n = max(4,int(sqrt(n)))
      fib_sequence = [0, 1]
      while len(fib_sequence)<n:
          fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
      return fib_sequence[2:]
    fib_seq = fibonacci_sequence(period)
    #print(f'fib_seq {fib_seq}')
    moving_averages = []
    for i in fib_seq:
        # Use np.convolve to calculate the moving average
        moving_avg = np.convolve(close, np.ones(i), 'valid') / i
        # Append zeros at the beginning to match the original array's size
        moving_avg = np.concatenate((np.zeros(i-1), moving_avg))
        moving_averages.append(moving_avg)
    # Calculate the average of the moving averages
    moving_averages = np.array(moving_averages)
    fma = np.mean(moving_averages, axis=0)
    return fma

#@feature_timeit
#@jit
def VAMA(close, volume, period):
    volume_weights = close * volume
    volume_weights_sum = np.convolve(volume_weights, np.ones(period), 'valid')
    volume_sum = np.convolve(volume, np.ones(period), 'valid')
    vama_values = volume_weights_sum / volume_sum
    return np.concatenate((np.full(period - 1, np.nan), vama_values))

#@feature_timeit
def anyMA_sig(np_close, np_xMA, np_ATR, atr_multi=1.000):
  #print(np_ATR)
  return ((np_xMA-np_close)/np_ATR)/atr_multi

######################################################################################
######################################################################################
######################################################################################

'''#@feature_timeit
def get_MA(np_df, type, MA_period):
  #ATR_multi=0.01
  #print(type, MA_period, ATR_period, ATR_multi)
  # Asumming np_df cols are following: O, H, L, C, Volume
  # (Volume index = 5)
  #np_df[:,1:5] = np_df[:,1:5].astype('float')
  #print(np_df)
  if type==0: return np.around(RMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==1: return np.around(talib.SMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==2: return np.around(talib.EMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==3: return np.around(talib.WMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==4: return np.around(VWMA(np_df[:,3], np_df[:,4], timeperiod=MA_period), 2)
  elif type==5: return np.around(talib.KAMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==6: return np.around(talib.TRIMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==7: return np.around(talib.DEMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==8: return np.around(talib.TEMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==9: return np.around(talib.T3(np_df[:,3], timeperiod=MA_period), 2)
  elif type==10: return np.around(finTA.SMM(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==11: return np.around(finTA.SSMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==12: return np.around(finTA.VAMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==13: return np.around(finTA.ZLEMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), max(4,MA_period)).to_numpy(), 2)
  elif type==14: return np.around(finTA.EVWMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==15: return np.around(finTA.SMMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==16: return np.around(finTA.HMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), MA_period).to_numpy(), 2)
  elif type==17: return np.around(ALMA(np_df[:,3], timeperiod=MA_period), 2)
  elif type==18: return np.around(HammingMA(np_df[:,3], MA_period), 2)
  elif type==19: return np.around(LSMA(np_df[:,3], max(3,MA_period)), 2)
  elif type==20: return np.around(LWMA(np_df[:,3],  MA_period), 2)
  elif type==21: return np.around(MGD(np_df[:,3], MA_period), 2)
  elif type==22: return np.around(VAMA(np_df[:,3], np_df[:,4], MA_period), 2)
  elif type==23: return np.around(GMA(np_df[:,3], MA_period), 2)
  elif type==24: return np.around(FBA(np_df[:,3], MA_period), 2)
  elif type==25: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=0), 2)
  elif type==26: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=1), 2)
  elif type==27: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=2), 2)
  elif type==28: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=3), 2)
  elif type==29: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=4), 2)
  elif type==30: return np.around(NadarayWatsonMA(np_df[:,3], MA_period, kernel=5), 2)
  elif type==31: return np.around(VIDYA(np_df[:,3], MA_period), 2)
  # testing
  #elif type==-1: return np.around(HullMA(np_df[:,3], timeperiod=max(4,MA_period)), 2)
  #elif type==-2: return np.around(finTA.FRAMA(pd.DataFrame(np_df, columns=['open', 'high', 'low', 'close', 'volume', 'X']), (lambda n: n + n % 2)(MA_period)).to_numpy(), 2)
'''
MA_TYPES = {0: lambda np_df,period: RMA(np_df[:,3], timeperiod=period),
            1: lambda np_df,period: talib.SMA(np_df[:,3], timeperiod=period),
            2: lambda np_df,period: talib.EMA(np_df[:,3], timeperiod=period),
            3: lambda np_df,period: talib.WMA(np_df[:,3], timeperiod=period),
            4: lambda np_df,period: VWMA(np_df[:,3], np_df[:,4], timeperiod=period),
            5: lambda np_df,period: talib.KAMA(np_df[:,3], timeperiod=period),
            6: lambda np_df,period: talib.TRIMA(np_df[:,3], timeperiod=period),
            7: lambda np_df,period: talib.DEMA(np_df[:,3], timeperiod=period),
            8: lambda np_df,period: talib.TEMA(np_df[:,3], timeperiod=period),
            9: lambda np_df,period: talib.T3(np_df[:,3], timeperiod=period),
            10: lambda np_df,period: finTA.SMM(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            11: lambda np_df,period: finTA.SSMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            12: lambda np_df,period: finTA.VAMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            13: lambda np_df,period: finTA.ZLEMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), max(4,period)).to_numpy(),
            14: lambda np_df,period: finTA.EVWMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            15: lambda np_df,period: finTA.SMMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            16: lambda np_df,period: finTA.HMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
            17: lambda np_df,period: ALMA(np_df[:,3], timeperiod=period),
            18: lambda np_df,period: HammingMA(np_df[:,3], period),
            19: lambda np_df,period: LSMA(np_df[:,3], max(3,period)),
            20: lambda np_df,period: LWMA(np_df[:,3],  period),
            21: lambda np_df,period: MGD(np_df[:,3], period),
            22: lambda np_df,period: VAMA(np_df[:,3], np_df[:,4], period),
            23: lambda np_df,period: GMA(np_df[:,3], period),
            24: lambda np_df,period: FBA(np_df[:,3], period),
            25: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=0),
            26: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=1),
            27: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=2),
            28: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=3),
            29: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=4),
            30: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=5),
            31: lambda np_df,period: VIDYA(np_df[:,3], period)
            }

#@feature_timeit
def get_MA(np_df, type, MA_period):
   return np.around(MA_TYPES[type](np_df, MA_period),2)

def get_MA_signal(np_df, type, MA_period, ATR_period, ATR_multi):
  atr = talib.ATR(np_df[:,1], np_df[:,2], np_df[:,3], ATR_period)
  return anyMA_sig(np_df[:,3],
                   get_MA(np_df, type, MA_period),
                   atr,
                   atr_multi=ATR_multi)

#@feature_timeit
def other_features(df: pd.DataFrame, suffix=''):
  _,O,H,L,C,V,*_= [df[col].to_numpy() for col in df.columns]
  df['RSI14'+suffix] = talib.RSI(C, timeperiod=14)
  df['RSI7'+suffix] = talib.RSI(C, timeperiod=7)
  df['RSI3'+suffix] = talib.RSI(C, timeperiod=3)
  df['ULT'+suffix] = talib.ULTOSC(H, L, C, timeperiod1=7, timeperiod2=14, timeperiod3=28)
  df['ADX'+suffix] = talib.ADX(H, L, C, timeperiod=14)
  df['-DI'+suffix] = talib.MINUS_DI(H, L, C, timeperiod=14)
  df['+DI'+suffix] = talib.PLUS_DI(H, L, C, timeperiod=14)
  df['MFI'+suffix] = talib.MFI(H, L, C, V, timeperiod=14)
  df['macd'+suffix],df['macdsignal'+suffix],df['macdhist'+suffix] = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
  df['ATR'+suffix] = talib.ATR(H, L, C, timeperiod=14)
  df['ADOSC'+suffix] = talib.ADOSC(H, L, C, V, fastperiod=3, slowperiod=10)
  df['APO'+suffix] = talib.APO(C, fastperiod=12, slowperiod=26, matype=0)
  df['AROONOSC'+suffix] =  talib.AROONOSC(H, L, timeperiod=14)
  df['STOCHRSIfastk'+suffix], df['STOCHRSIfastd'+suffix] = talib.STOCHRSI(C, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
  df['CMO'+suffix] = talib.CMO(C, timeperiod=14)
  df['BOP'+suffix] = talib.BOP(O, H, L, C)
  df['TRANGE'+suffix] = talib.TRANGE(H, L, C)
  df['PPO'+suffix] = talib.PPO(C, fastperiod=12, slowperiod=26, matype=0)
  df['WILLR'+suffix] = talib.WILLR(H, L, C, timeperiod=14)
  df['KST'+suffix] = ta.trend.kst_sig(df['Close'])
  df['Vortex'+suffix] = ta.trend.VortexIndicator(df['High'], df['Low'], df['Close']).vortex_indicator_diff()
  df['STC'+suffix] = ta.trend.STCIndicator(df['Close']).stc()
  df['PVO'+suffix] = ta.momentum.PercentageVolumeOscillator(df['Volume']).pvo()
  df['AO'+suffix] =  ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low']).awesome_oscillator()
  df['up_x1'], df['mid'+suffix], df['low_x1'] = talib.BBANDS(C, timeperiod=11, nbdevup=1.5, nbdevdn=1.5, matype=0)
  df['up_x2'], _, df['low_x2'] = talib.BBANDS(C, timeperiod=11, nbdevup=2.0, nbdevdn=2.0, matype=0)
  df['up_x3'], _, df['low_x3'] = talib.BBANDS(C, timeperiod=11, nbdevup=2.5, nbdevdn=2.5, matype=0)
  df['BB_mid_dist'] = df['mid'+suffix]-C
  ### Signals 
  df['RSI3_sig'+suffix] = ULT_RSI_signal(df['RSI3'+suffix], timeperiod=3)
  df['RSI7_sig'+suffix] = ULT_RSI_signal(df['RSI7'+suffix], timeperiod=7)
  df['RSI14_sig'+suffix] = ULT_RSI_signal(df['RSI14'+suffix], timeperiod=14)
  df['ULT_sig'+suffix] = ULT_RSI_signal(df['ULT'+suffix], timeperiod=28)
  df['ADX_sig'+suffix] = ADX_signal(df['ADX'+suffix], df['-DI'+suffix], df['+DI'+suffix])
  df['ADX_trend_sig'+suffix] = ADX_trend(df['ADX'+suffix])
  df['MFI_sig'+suffix] = MFI_signal(df['MFI'+suffix])
  df['MFI_divergence_sig'+suffix] = MFI_divergence(df['MFI'+suffix], df['Close'])
  df['MACD_cross_sig'+suffix] = MACD_cross(df['macd'+suffix], df['macdsignal'+suffix])
  df['MACDhist_reversal_sig'+suffix] = MACDhist_reversal(df['macdhist'+suffix])
  df['MACD_zerocross_sig'+suffix] = MACD_zerocross(df['macd'+suffix],df['macdsignal'+suffix])
  df['BB_sig'+suffix] = BB_sig(df['mid'+suffix].to_numpy(), df['up_x1'].to_numpy(), df['low_x1'].to_numpy(), df['up_x2'].to_numpy(), df['low_x2'].to_numpy(), df['up_x3'].to_numpy(), df['low_x3'].to_numpy(), C)
  return df

def signal_only_features(df: pd.DataFrame, suffix=''):
  _,O,H,L,C,V,*_= [df[col].to_numpy() for col in df.columns]
  ### Signals 
  df['RSI3_sig'+suffix] = ULT_RSI_signal(talib.RSI(C, timeperiod=3), timeperiod=3)
  df['RSI7_sig'+suffix] = ULT_RSI_signal(talib.RSI(C, timeperiod=7), timeperiod=7)
  df['RSI14_sig'+suffix] = ULT_RSI_signal(talib.RSI(C, timeperiod=14), timeperiod=14)
  df['ULT_sig'+suffix] = ULT_RSI_signal(talib.ULTOSC(H, L, C, timeperiod1=7, timeperiod2=14, timeperiod3=28), timeperiod=28)
  df['ADX_sig'+suffix] = ADX_signal(talib.ADX(H, L, C, timeperiod=14), talib.MINUS_DI(H, L, C, timeperiod=14), talib.PLUS_DI(H, L, C, timeperiod=14))
  df['ADX_trend_sig'+suffix] = ADX_trend(talib.ADX(H, L, C, timeperiod=14))
  df['MFI_sig'+suffix] = MFI_signal(talib.MFI(H, L, C, V, timeperiod=14))
  df['MFI_divergence_sig'+suffix] = MFI_divergence(talib.MFI(H, L, C, V, timeperiod=14), C)
  MACD, MACD_sig, MACD_hist = talib.MACD(C, fastperiod=12, slowperiod=26, signalperiod=9)
  df['MACD_cross_sig'+suffix] = MACD_cross(MACD, MACD_sig)
  df['MACDhist_reversal_sig'+suffix] = MACDhist_reversal(MACD_hist)
  df['MACD_zerocross_sig'+suffix] = MACD_zerocross(MACD, MACD_sig)
  up_x1, mid, low_x1 = talib.BBANDS(C, timeperiod=11, nbdevup=1.5, nbdevdn=1.5, matype=0)
  up_x2, _, low_x2 = talib.BBANDS(C, timeperiod=11, nbdevup=2.0, nbdevdn=2.0, matype=0)
  up_x3, _, low_x3 = talib.BBANDS(C, timeperiod=11, nbdevup=2.5, nbdevdn=2.5, matype=0)
  df['BB_sig'+suffix] = BB_sig(mid, up_x1, low_x1, up_x2, low_x2, up_x3, low_x3, C)
  df['Volume_probablity'] = Volume_probablity(V)
  df['Move_probablity'] = Move_probablity(O,C)
  df['SUM'] = df.iloc[:, 6:].sum(axis=1)
  return df

def basic_features(df, suffix=''):
  _,O,H,L,C,V,*_= [df[col].to_numpy() for col in df.columns]
  # OHLC simple features
  df['candle_size'+suffix] = H-L
  df['candle_body_size'+suffix] = np.where(C>O, (C-O)/df['candle_size'+suffix], (O-C)/df['candle_size'+suffix])
  df['candle_upper_wick'+suffix] = np.where(C>O, (H-C)/df['candle_size'+suffix], (H-O)/df['candle_size'+suffix])
  df['candle_lower_wick'+suffix] = np.where(C>O, (O-L)/df['candle_size'+suffix], (C-L)/df['candle_size'+suffix])
  df['Hourly_seasonality'] = Hourly_seasonality(df)
  df['Daily_seasonality'] = Daily_seasonality(df)
  df['Volume_probablity'] = Volume_probablity(V)
  df['Move_probablity'] = Move_probablity(O,C)
  df['Price_levels'] = Price_levels(O,C)
  return df

def blank_features(df, *args, **kwargs):
  return df

TA_FEATURES_TEMPLATE = {'None':blank_features,
                        'basic':basic_features,
                        'signals':signal_only_features}

def get_combined_intervals_df(ticker, interval_list, type='um', data='klines', template='None'):
  for itv in interval_list:
    if itv==interval_list[0] or len(interval_list)==1:
      df = get_data.by_BinanceVision(ticker=ticker, interval=itv, type=type, data=data)
      df = TA_FEATURES_TEMPLATE[template](df, ticker)
    else:
      _df = get_data.by_BinanceVision(ticker=ticker, interval=itv, type=type, data=data)
      _df['Opened'] = _df['Opened'] + (_df.iloc[-1]['Opened']-_df.iloc[-2]['Opened'])
      _df = TA_FEATURES_TEMPLATE[template](_df, ticker, suffix='_'+itv)
      df = pd.merge_asof(df, _df, on='Opened', direction='backward', suffixes=('', '_'+itv))
  df.fillna(method='ffill', inplace=True)
  return df