#from os import environ
#environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from time import time
from statistics import mean, stdev
from math import sqrt, copysign
from finta import TA as finTA
from tindicators import ti
import talib
import ta 
import get_data
from utility import linear_reg_slope
from numba import jit, njit
#config.THREADING_LAYER = 'safe'
#import MAs
#import datetime as dt
#from numba import njit, jit
#from scipy.ndimage import convolve1d as conv
#from sklearn import preprocessing

def feature_timeit(feature_func: callable) -> callable:
  """
  Measure the execution time of a TA feature function.

  Args:
      feature_func (Callable): The feature function to be timed.
  Returns:
      Callable: A wrapped version of the feature function that measures and
      prints the execution time.
  """
  def wrapper(*args, **kwargs):
    start_t = time()
    print(f'\r adding {feature_func.__name__} feature...', end='')
    ret = feature_func(*args, **kwargs)
    print(f' ({(time()-start_t):.3f}s)')
    return ret
  return wrapper

@feature_timeit
def scaleColumns(df: pd.DataFrame, scaler: callable) -> pd.DataFrame:
    """
    Scales TA indicators inside provided dataframe.

    Args:
        df (pd.DataFrame): dataframe of TA indicators values to scale 
        scaler (callable): sklearn preprocessing scaler to be used ex. MinMaxScaler()
    Returns:
        pd.DataFrame: scaled dataframe

    It skips OHLC and date/time columns.
    """
    for col in df.columns:
        if col not in ['Open time', 'Opened', 'Close time', 'Open', 'High', 'Low', 'Close']:
            #print(col)
            #caler.fit(df[[col]])
            df[col] = scaler.fit_transform(df[[col]])
    return df

def linear_slope_indicator(self, values: np.ndarray | list) -> float:
      """
      Computes a linear slope indicator based on a subset of values.

      Args:
          values (np.ndarray | list): Input values for computing the slope indicator.
          This can be either a NumPy array or a list of numerical values.

      Returns:
          float: The computed linear slope indicator.

      The function computes the linear slope indicator using a subset of the input values. 
      It calculates the slope/s using a specified portions of the input values and then 
      applies a some transformation to the result before returning it.

      Example:
          >>> values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
          >>> indicator = linear_slope_indicator(values)
          >>> print(indicator)
          1.0
      """
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
@jit(nopython=True)
def RSI_like_signal(RSIlike_indicator: np.ndarray | list[float], timeperiod: int) -> list[float]:
  """
  Calculate RSI like (Ultimate oscillator, MFI) signals for a given column of values.

  Args:
      RSIlike_indicator (np.ndarray or List[float]): An array or list of RSI like values [0,100].
      timeperiod (int): The time period for calculating the signal.

  Returns:
      List[float]: A list of signals corresponding to the input values [-1,1].

  Input values interpretation:
    >=90, singal: -0.75, extremely overbought
    >=75, signal: -0.5, overbought
    >=65, signal: -0.25, somewhat overbought
    <=10, signal: 0.75, extremely oversold
    <=25, signal: 0.5, oversold
    <=35, signal: 0.25, somewhat oversold
    For 80/20 line crosses:
    80.0 line cross from above, signal: -1, bearish cross
    20.0 line cross from below, signal: 1, bullish cross 

  Example:
      >>> rsi_values = [60.0, 55.0, 45.0, 35.0, 19.0, 40.0, 75.0, 85.0]
      >>> RSIlike_signal(rsi_values, 3)
      [None, None, None, 0.25, 0.5, 1, -0.25, -0.5]
  """
  return [0.0] * timeperiod + [  -1 if (prev > 80.0) and (cur <= 80.0) else
                                  1 if (prev < 20.0) and (cur >= 20.0) else
                                  -.75 if (cur > 90.0) else
                                  .75 if (cur < 10.0) else
                                  -.5 if (cur > 75.0) else
                                  .5 if (cur < 25.0) else
                                  -.25 if (cur > 65.0) else
                                  .25 if (cur < 35.0) else
                                  0
                                  for cur, prev in zip(RSIlike_indicator[timeperiod:], RSIlike_indicator[timeperiod-1:-1])  ]

@feature_timeit
def ADX_signal(adx_col: np.ndarray | list, minus_DI: np.ndarray | list, plus_DI: np.ndarray | list) -> list[float]:
  """
  Calculate ADX (Average Directional Index) signals based on the given parameters.

  Args:
      adx_col (np.ndarray or List[float]): An array of ADX values.
      minus_DI (np.ndarray or List[float]): An array of minus DI values.
      plus_DI (np.ndarray or List[float]): An array of plus DI values.

  Returns:
      List[float]: A list of ADX signals calculated according to the specified conditions:
      
      0.0 when adx is below 25.0
      1.0 buliish DI cross and adx above 35.0
      0.75 same but adx only above 25.0
      -1.0 bearish DI cross and adx above 25.0
      -0.75 same but adx only above 25.0
      0.5 DIs are getting closer to bullish cross
      -0.5 DIs are getting closer to bearish cross
      0.25 getting closer to bullish cross but with adx only above 25.0:
      -0.25 getting closer to bearish cross but with adx only above 25.0:
      0.0 for other cases

  Note:
      The input arrays must have the same length. 
      First value in returned list is always 0.0

  Example:
      >>> adx_col = np.array([30.0, 28.0, 26.0, 25.0])
      >>> minus_DI = np.array([10.0, 12.0, 14.0, 16.0])
      >>> plus_DI = np.array([20.0, 18.0, 16.0, 14.0])
      >>> ADX_signal(adx_col, minus_DI, plus_DI)
      [0.0, -0.5, -0.5, -1.0]
  """
  return [0.0] + [  0.0 if adx<25.0 else
                    1 if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI) and (adx >= 35.0) else
                    .75 if (cur_pDI > cur_mDI) and (prev_pDI < prev_mDI) and (adx >= 25.0) else
                    -1 if (cur_pDI < cur_mDI) and (prev_pDI > prev_mDI) and (adx >= 35.0) else
                    -.75 if (cur_pDI < cur_mDI) and (prev_pDI > prev_mDI) and (adx >= 25.0) else
                    .5 if (cur_pDI > prev_pDI) and (cur_mDI < prev_mDI) and (cur_pDI < cur_mDI) and (adx >= 35.0) else
                    -.5 if (cur_pDI < prev_pDI) and (cur_mDI > prev_mDI) and (cur_pDI > cur_mDI) and (adx >= 35.0) else
                    .25 if (cur_pDI > prev_pDI) and (cur_mDI < prev_mDI) and (cur_pDI < cur_mDI) and (adx >= 25.0) else
                    -.25 if (cur_pDI < prev_pDI) and (cur_mDI > prev_mDI) and (cur_pDI > cur_mDI) and (adx >= 25.0) else
                    0.0
                    for cur_pDI, cur_mDI, adx, prev_pDI, prev_mDI in zip(plus_DI[1:], minus_DI[1:], adx_col[1:], plus_DI[:-1], minus_DI[:-1]) ]

@feature_timeit
def ADX_trend_signal(adx_col: np.ndarray | list, minus_DI: np.ndarray | list, plus_DI: np.ndarray | list) -> list[float | int]:
  """
  Calculates the ADX trend signal based on the given ADX, minus DI, and plus DI values.

  Args:
      adx_col (np.ndarray or List[float]): An array or list of ADX values.
      minus_DI (np.ndarray or List[float]): An array or list of minus DI values.
      plus_DI (np.ndarray or List[float]): An array or list of plus DI values.

  Returns:
      List[float]: A list of trend signals. Possible values:
          1: Strong uptrend (ADX >= 75.0 and plus DI > minus DI)
          -1: Strong downtrend (ADX >= 75.0 and plus DI < minus DI)
          0.75: Moderate uptrend (ADX >= 50.0 and plus DI > minus DI)
          -0.75: Moderate downtrend (ADX >= 50.0 and plus DI < minus DI)
          0.5: Weak uptrend (ADX >= 25.0 and plus DI > minus DI)
          -0.5: Weak downtrend (ADX >= 25.0 and plus DI < minus DI)
          0.25: Very weak uptrend (ADX < 25.0 and plus DI > minus DI)
          -0.25: Very weak downtrend (ADX < 25.0 and plus DI < minus DI)
          0: No clear trend.

  Note:
      The function calculates the trend signal for each set of corresponding ADX, minus DI, and plus DI values
      in the input arrays or lists.

  Example:
      >>> ADX = [70.0, 80.0, 60.0]
      >>> minusDI = [20.0, 30.0, 40.0]
      >>> plusDI = [40.0, 50.0, 30.0]
      >>> ADXtrend_signal(adx_values, minus_di_values, plus_di_values)
      [0.75, 1.0, -0.75]
  """
  return [  1 if (adx >= 75.0) and (pDI > mDI) else
            -1 if (adx >= 75.0) and (pDI < mDI) else
            .75 if (adx >= 50.0) and (pDI > mDI) else
            -.75 if (adx >= 50.0) and (pDI < mDI) else
            .5 if (adx >= 25.0) and (pDI > mDI) else
            -.5 if (adx >= 25.0) and (pDI < mDI) else
            .25 if (adx < 25.0) and (pDI > mDI) else
            -.25 if (adx < 25.0) and (pDI < mDI) else
            0 
            for adx, mDI, pDI in zip(adx_col, minus_DI, plus_DI) ]

@feature_timeit
def MACD_cross_signal(macd_col: np.ndarray | list, signal_col: np.ndarray | list) -> list[float | int]:
  """
  Calculate MACD (Moving Average Convergence Divergence) crossover signals.

  Args:
      macd_col (np.ndarray or List[float]): An array or list containing MACD values.
      signal_col (np.ndarray or List[float]): An array or list containing signal line values.

  Returns:
      List[float]: A list of crossover signals with values 1, -1, 0.5, or -0.5.

  The function calculates crossover signals based on the provided MACD and signal data. It compares
  the current MACD and signal values with the previous values and assigns a signal.

  Note:
  The input arrays or lists should have the same length.

  Example:
  >>> macd_values = [12.5, 14.2, 11.8, 15.6, 9.8]
  >>> signal_values = [10.3, 12.6, 11.2, 13.9, 10.3]
  >>> signals = MACD_cross(macd_values, signal_values)
  >>> print(signals)
  [0, 0, 0, 0, -1]
  """
  return [0.0] + [  1 if (cur_sig < 0) and (cur_macd < 0) and (cur_macd>cur_sig) and (prev_macd<prev_sig) else
                    .75 if (cur_macd>cur_sig) and (prev_macd<prev_sig) else
                    -1 if (cur_sig > 0) and (cur_macd > 0) and cur_macd<cur_sig and prev_macd>prev_sig else
                    -.75 if cur_macd<cur_sig and prev_macd>prev_sig else
                    0.5 if cur_macd>prev_macd and cur_sig<prev_sig else
                    -0.5 if cur_macd<prev_macd and cur_sig>prev_sig else
                    0
                    for cur_sig, cur_macd, prev_sig, prev_macd in zip(signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1]) ]

@feature_timeit
def MACDhist_reversal_signal(macdhist_col: np.ndarray | list) -> list[float | int]:
  """
  Calculate MACD histogram reversal signals.

  Args:
      macdhist_col (np.ndarray or List[float]): An array or list of MACD histogram values.

  Returns:
      List[float]: A list of reversal signals, where 1 represents a bullish reversal,
      -1 represents a bearish reversal, and 0 represents no reversal.

  The function calculates MACD histogram reversal signals based on the MACD histogram values provided in the input.
  It compares the current MACD histogram value with the previous three values to determine whether a bullish
  or bearish reversal has occurred.

  Note:
  - If cur_macd > prev_macd and prev_macd < preprev_macd < prepreprev_macd, it's considered a bullish reversal (returns 1).
  - If cur_macd < prev_macd and prev_macd > preprev_macd > prepreprev_macd, it's considered a bearish reversal (returns -1).
  - Otherwise, no reversal is detected (returns 0).

  Example:
  >>> macdhist_values = [0.5, 0.6, 0.4, 0.3, 0.8, 0.9, 1.0, 0.7]
  >>> signals = MACDhist_reversal(macdhist_values)
  >>> print(signals)
  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0]
  """
  return [0.0] * 3 + [  1.0 if cur_macd>prev_macd and prev_macd<preprev_macd<prepreprev_macd else
                        -1.0 if cur_macd<prev_macd and prev_macd>preprev_macd>prepreprev_macd else
                        0
                        for cur_macd, prev_macd, preprev_macd, prepreprev_macd in zip(macdhist_col[3:], macdhist_col[2:-1], macdhist_col[1:-2], macdhist_col[:-3]) ]

@feature_timeit
def MACD_zerocross_signal(macd_col: np.ndarray | list, signal_col: np.ndarray | list) -> list[float | int]:
  """
  Calculate the zero crossing values for MACD (Moving Average Convergence Divergence).

  This function takes two lists or arrays, `macd_col` and `signal_col`, and computes zero crossing values for MACD.
  
  Zero crossing is determined based on the relationship between current and previous values in both columns.

  Args:
      macd_col (np.ndarray or List[float]): An array or list containing MACD values.
      signal_col (np.ndarray or List[float]): An array or list containing signal line values.

  Returns:
      List[Union[float, int]]: A list of zero crossing singals [-1,1]

  Signals are generated on the following criteria:
    - 0.5 for MACD going from negative to positive
    - 1 for Signal line going from negative to positive
    - 0.5 for MACD going from positive to negative
    - -1 for Signal line going from positive to negative
    - 0 for no zero crossing

  Note:
      - The length of `macd_col` and `signal_col` should be the same.
      - The function assumes that both lists have at least 2 elements.

  Examples:
      >>> macd = [1.0, 0.5, -0.2, -0.7, 0.9]
      >>> signal = [0.5, 0.1, -0.3, -0.6, 0.8]
      >>> MACD_zerocross(macd, signal)
      [0, 0.5, -0.5, 0, 1]

  """
  return [0.0] + [  1 if cur_sig>0 and prev_sig<0 else
                    -1 if cur_sig<0 and prev_sig>0 else
                    .5 if cur_macd>0 and prev_macd<0 else
                    -.5 if cur_macd<0 and prev_macd>0 else
                    0
                    for cur_macd, cur_sig, prev_macd, prev_sig in zip(macd_col[1:], signal_col[1:], macd_col[:-1], signal_col[:-1]) ]

@feature_timeit
def BB_signal(close: np.ndarray | list,
              upper_band: np.ndarray | list,
              mid: np.ndarray | list,
              lower_band: np.ndarray | list) -> list[int]:
  """
  Calculate Bollinger Bands signal based on the provided data.

  Args:
      close (np.ndarray or List[float]): A one-dimensional array of closing prices.
      upper_band (np.ndarray or List[float]): A one-dimensional array of upper Bollinger Bands values.
      mid (np.ndarray or List[float]): A one-dimensional array of middle Bollinger Bands values.
      lower_band (np.ndarray or List[float]): A one-dimensional array of lower Bollinger Bands values.

  Returns:
      List[int]: A list of signals ranging from strong buy (1) to strong sell (-1).

  Note:
      - Buy (1) when the current closing price is below the lower Bollinger Band.
      - Weak buy (0.5) when the current closing price crossed mid from below.
      - Sell (-1) when the current closing price is above the upper Bollinger Band.
      - Weak sell (-0.5) when the current closing price crossed mid from above.
      - Hold (0) in other cases.

  The function calculates Bollinger Bands signals for each pair of consecutive data points and
  returns a list of signals.

  Example:
      close_prices = [50.0, 52.0, 48.0, 56.5, 57.0]
      upper_bands = [52.0, 54.0, 50.0, 57.0, 55.0]
      mid_bands = [51.0, 53.0, 49.0, 56.0, 54.0]
      lower_bands = [50.0, 52.0, 48.0, 55.0, 53.0]

      signals = BB_signal(close_prices, upper_bands, mid_bands, lower_bands)
      # Output: [1, 1, 1, 0.5, -1]

  """
  return [0.0] + [  -1 if currC >= Ub else
                    1 if currC <= Lb else
                    -.5 if (currC < Mid) and (preC > Mid) else
                    .5 if (currC > Mid) and (preC < Mid) else
                    0
                    for currC,preC,Lb,Mid,Ub in zip(close[1:],close[:-1],lower_band[1:],mid[1:],upper_band[1:])  ]

@feature_timeit
def Price_levels(Open: np.ndarray | list, Close: np.ndarray | list, decimals: int=0, sort: bool=False) -> list[int]:
  """
  Calculate price levels based on Open and Close prices.

  Args:
      Open (np.ndarray | List[float]): An array or list of opening prices.
      Close (np.ndarray | List[float]): An array or list of closing prices.
      decimals (int, optional): Number of decimal places to round prive-level to. Defaults to 0.
      sort (bool, optional): Whether to sort the price-levels by frequency. Defaults to False.

  Returns:
      List[int]: A list of integer values representing the frequency of price levels.

  The function calculates price levels based on the Open and Close prices. Price levels are determined
  by checking if the price moved up or down between two consecutive periods and rounding the Close
  price to the specified number of decimals. The resulting price-levels are then counted, and the
  function returns a list of integers representing the frequency of each price level.

  If `sort` is set to True, the price levels are sorted in descending order of frequency.
  """
  def get_levels(open, close, decimals=0, sort=False):
    tckr_lvls={}
    for open, close, close_next in zip(open[:-1], close[:-1], close[1:]):
      if (close>open>close_next) or (close<open<close_next):
        lvl = round(close, decimals)
        if lvl in tckr_lvls: tckr_lvls[lvl] += 1
        else: tckr_lvls[lvl] = 1
    if sort:
       tckr_lvls = { k:v for k,v in sorted(tckr_lvls.items(), key=lambda item: item[1], reverse=True) }
    return tckr_lvls
  
  lvls = get_levels(Open, Close, decimals=decimals, sort=sort)
  return [ lvls[round(c, decimals)] if round(c, decimals) in lvls.keys() else 0 for c in Close ]

@feature_timeit
def Move_probablity(Open: np.ndarray | list, Close: np.ndarray | list) -> list[float]:
  """
  Calculate the probability of a price move at given moment based on distribution (by standardizing) of historical Open and Close data.

  Args:
      Open (np.ndarray or list[float]): An array or list of opening prices.
      Close (np.ndarray or list[float]): An array or list of closing prices.

  Returns:
      List[float]: A list of move probabilities for each data point.

  The function calculates the probability of a move by comparing the Open and Close prices.
  It uses the following formula:

  probability = ((price_change - avg_gain) / gain_stdev) if close > open,
                  else ((price_change - avg_loss) / loss_stdev)

  where price_change is the percentage change in price, avg_gain is the average gain,
  gain_stdev is the standard deviation of gains, avg_loss is the average loss, and
  loss_stdev is the standard deviation of losses.

  Note: This function assumes that Open and Close arrays/lists have the same length.
  """
  def get_avg_changes(open, close):
    gain = [ (close/open-1)*100 for open,close in zip(open,close) if close>open ]
    loss = [ (open/close-1)*100 for open,close in zip(open,close) if close<open ]
    return mean(gain),stdev(gain),mean(loss),stdev(loss)
  
  avg_gain, gain_stdev, avg_loss, loss_stdev = get_avg_changes(Open, Close)
  return [ (((close/open-1)*100)-avg_gain)/gain_stdev if close>open else (((open/close-1)*100)-avg_loss)/loss_stdev for open,close in zip(Open,Close) ]

@feature_timeit
def Volume_probablity(Volume: np.ndarray | list) -> np.ndarray:
  """
  Calculate the probability distribution of a given volume data.

  This function takes a list or NumPy array containing volume data and calculates the probability distribution
  by standardizing the data. It subtracts the mean and divides by the standard deviation.

  Args:
      Volume (np.ndarray or list): A list or NumPy array containing volume data.

  Returns:
      np.ndarray: A NumPy array representing the probability distribution of the input volume data.
  """
  return (Volume-np.mean(Volume))/np.std(Volume)

@feature_timeit
def Hourly_seasonality_by_ticker(ticker: str, Opened_dt_hour: pd.Series, type: str='um', data: str='klines') -> list[float]:
  """
  Calculate hourly seasonality for a given ticker.

  Args:
      ticker (str): The ticker symbol. (e.g., 'BTCUSDT')
      Opened_dt_hour (pd.Series): A pandas Series containing the hour for each data point. [0,24)
      type (str, optional): The type of data 'um' (USDs-M) or 'cm' (COIN-M) for futures, 'spot' for spot. Defaults to 'um'.
      data (str, optional): The data source: {aggTrades, bookDepth, bookTicker, indexPriceKlines, klines, liquidationSnapshot, markPriceKlines, metrics, premiumIndexKlines, trades}.
                            Defaults to 'klines'.

  Returns:
      List[float]: A list of hourly seasonality values for the given ticker.

  This function calculates the hourly seasonality for a given ticker based on the provided hour series. 
  It first retrieves data using the specified parameters and then computes the average percentage change for each hour of the day (0-23).
  Finally, it returns a list of hourly seasonality values corresponding to the provided opening hours.
  """
  def get_hour_changes(ticker='BTCUSDT', type='um', data='klines'):
    tckr_df = get_data.by_BinanceVision(ticker, '1h', type=type, data=data)
    hour = tckr_df['Opened'].dt.hour.to_numpy()
    CO_chng = ((tckr_df['Close'].to_numpy()/tckr_df['Open'].to_numpy())-1)*100
    return { i:np.mean([chng for chng, h in zip(CO_chng, hour) if h==i]) for i in range(0,24) }
  
  h_dict = get_hour_changes(ticker=ticker, type=type, data=data)
  return [h_dict[h] for h in Opened_dt_hour]

@feature_timeit
def Hourly_seasonality(df: pd.DataFrame) -> list[float]:
  """
  Calculates the hourly seasonality of stock price changes.

  Args:
      df (pd.DataFrame): A DataFrame containing price data.
          It should have a 'Opened' column with datetime values and 'Close' and 'Open' column
          with corresponding prices.

  Returns:
      List[float]: A list of hourly seasonality values, representing the average
          percentage change in stock price for each hour of the day (0-23).
  """
  hour = df['Opened'].dt.hour
  CO_chng = ((df['Close']/df['Open'])-1)*100
  # get mean change(chng) for each hour(i) in day
  h_dict = { i:np.mean([chng for chng, h in zip(CO_chng, hour) if h==i]) for i in range(0,24) }
  return [ h_dict[h] for h in hour ]

@feature_timeit
def Daily_seasonality_by_ticker(ticker: str, Opened_dt_weekday: pd.Series, type='um', data='klines') -> list[float]:
  """
  Compute the daily seasonality for a given ticker.

  Args:
      ticker (str): The ticker symbol, e.g., 'BTCUSDT'.
      Opened_dt_weekday (pd.Series): A Pandas Series containing the weekdays for each data point. [0,7)
      type (str, optional): The type of data 'um' (USDs-M) or 'cm' (COIN-M) for futures, 'spot' for spot. Defaults to 'um'.
      data (str, optional): The data source: {aggTrades, bookDepth, bookTicker, indexPriceKlines, klines, liquidationSnapshot, markPriceKlines, metrics, premiumIndexKlines, trades}.
                            Defaults to 'klines'.

  Returns:
      List[float]: A list of daily seasonality values corresponding to each weekday.

  Example:
  >>> ticker = 'BTCUSDT'
  >>> weekdays = pd.Series([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])  # Example weekday data
  >>> seasonality = Daily_seasonality_by_ticker(ticker, weekdays)

  Note:
  - This function calculates the daily seasonality as the mean change (in percentage) from Open to Close
    for a given ticker, grouped by each day of the week (0 to 6, where 0 is Monday and 6 is Sunday).
  - The function internally uses the `get_data.by_BinanceVision` function to retrieve data for the ticker.

  """
  def get_weekday_changes(ticker='BTCUSDT', type='um', data='klines'):
    tckr_df = get_data.by_BinanceVision(ticker, '1d', type=type, data=data)
    weekday = tckr_df['Opened'].dt.dayofweek.to_numpy()
    CO_chng=((tckr_df['Close'].to_numpy()/tckr_df['Open'].to_numpy())-1)*100
    return { i:np.mean([chng for chng, day in zip(CO_chng, weekday) if day==i]) for i in range(0,7) }
  wd_dict = get_weekday_changes(ticker=ticker, type=type, data=data)
  return [ wd_dict[w] for w in Opened_dt_weekday ]

@feature_timeit
def Daily_seasonality(df: pd.DataFrame) -> list[float]:
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
def HullMA(close: np.ndarray | list, timeperiod: int) -> pd.Series:
  return talib.WMA( (talib.WMA(close, timeperiod//2 )*2)-(talib.WMA(close, timeperiod)), int(np.sqrt(timeperiod)) )

#@feature_timeit
@jit(nopython=True)
def RMA(close: np.ndarray, timeperiod: int) -> np.ndarray[np.float64]:
  alpha = 1.0 / timeperiod
  #rma = [0.0] * len(close)
  rma = np.zeros_like(close)
  # Calculating the SMA for the first 'length' values
  sma = sum(close[:timeperiod]) / timeperiod
  rma[timeperiod - 1] = sma
    # Calculating the rma for the rest of the values
  for i in range(timeperiod, len(close)):
    rma[i] = alpha * close[i] + (1 - alpha) * rma[i - 1]
  return rma

#@feature_timeit
@jit(nopython=True)
def VWMA(close: np.ndarray, volume: np.ndarray, timeperiod: int) -> list[float]:
    cum_sum = 0
    cum_vol = 0
    vwmas = []
    cv_list = close*volume
    i = 0
    while i<len(close):
        cum_sum += cv_list[i]
        cum_vol += volume[i]
        if i >= timeperiod:
            cum_sum -= cv_list[i - timeperiod]
            cum_vol -= volume[i - timeperiod]
        vwmas.append(cum_sum / cum_vol)
        i += 1
    return vwmas

#@feature_timeit
def ALMA(close: np.ndarray, timeperiod: int, offset: float=0.85, sigma: int=6) -> np.ndarray[np.float64]:
    m = offset * (timeperiod - 1)
    s = timeperiod / sigma
    wtd = np.array([np.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(timeperiod)])
    wtd /= sum(wtd)
    alma = np.convolve(close, wtd, mode='valid')
    return np.insert(alma, 0, [np.nan]*(timeperiod-1))

#@feature_timeit
def HammingMA(close: np.ndarray, timeperiod: int) -> np.ndarray[np.float64]:
    w = np.hamming(timeperiod)
    hma = np.convolve(close, w, mode='valid') / w.sum()
    return np.insert(hma, 0, [np.nan]*(timeperiod-1))

@jit(nopython=True)
def LSMA(close: np.ndarray, timeperiod: int) -> np.ndarray[np.float64]:
 close = np.ascontiguousarray(close)
 #n = len(close)
 lsma = np.empty_like(close)
 lsma[:timeperiod-1] = np.nan
 # Przygotuj x raz i używaj w pętli
 x = np.arange(0, timeperiod)
 A = np.empty((timeperiod, 2))
 A[:, 0] = x
 A[:, 1] = 1
 AT = np.ascontiguousarray(A.T)
 ATA_inv = np.linalg.inv(np.dot(AT, A))
 for i in range(timeperiod - 1, len(close)):
     y = close[i - timeperiod + 1:i + 1]
     m, c = np.dot(ATA_inv, np.dot(AT, y))
     lsma[i] = m * (timeperiod-1) + c
 return lsma

@jit(nopython=True)
def gaussian_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
@jit(nopython=True)
def epanechnikov_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.where(np.abs(x) <= 1, 3/4 * (1 - x**2), 0)
@jit(nopython=True)
def rectangular_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.where(np.abs(x) <= 1, 0.5, 0)
@jit(nopython=True)
def triangular_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.where(np.abs(x) <= 1, 1 - np.abs(x), 0)
@jit(nopython=True)
def biweight_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.where(np.abs(x) <= 1, (15/16) * (1 - x**2)**2, 0)
@jit(nopython=True)
def cosine_kernel(x: np.ndarray) -> np.ndarray[np.float64]:
    return np.where(np.abs(x) <= 1, np.pi/4 * np.cos(np.pi/2 * x), 0)
@jit(nopython=True)
def NadarayWatsonMA(close: np.ndarray, timeperiod: int, kernel: int=0) -> np.ndarray[np.float64]:
    nwma = np.empty_like(close)
    nwma[:timeperiod-1] = np.nan
    distances = np.abs(np.arange(timeperiod) - (timeperiod-1))
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
    weights = np.ascontiguousarray(weights)
    for i in range(timeperiod-1, len(close)):
        window_prices = np.ascontiguousarray(close[i-timeperiod+1:i+1])
        nwma[i] = (weights @ window_prices) / weights.sum()
    #nwma = nwma[:timeperiod]+[ (weights @ np.ascontiguousarray(close[i-timeperiod+1:i+1])) / weights.sum() for i in range(timeperiod-1, len(close)) ]
    return nwma

#@feature_timeit
@jit(nopython=True)
def LWMA(close: np.ndarray, period: int) -> np.ndarray[np.float64]:
    weights = np.arange(1, period + 1).astype(np.float64)
    close = np.ascontiguousarray(close)
    lwma = np.zeros_like(close)
    for i in range(period - 1, len(close)):
        lwma[i] = np.dot(weights, close[i - period + 1 : i + 1]) / weights.sum()
    return lwma

#@feature_timeit
@jit(nopython=True)
def MGD(close: np.ndarray, period: int) -> np.ndarray[np.float64]:
    md = np.zeros_like(close)
    md[0] = close[0]
    for i in range(1, len(close)):
        md[i] = md[i-1] + (close[i] - md[i-1]) / (period * np.power((close[i] / md[i-1]), 4))
    return md

### It behaves differently depending on close len
#@feature_timeit
@jit(nopython=True)
def VIDYA(close: np.ndarray, k: np.ndarray, period: int) -> np.ndarray[np.float64]:
    alpha = 2 / (period + 1)
    #k = talib.CMO(close, period)
    k = np.abs(k)/100
    VIDYA = np.zeros_like(close)
    VIDYA[period-1] = close[period-1]
    for i in range(period, len(close)):
        VIDYA[i] = alpha * k[i] * close[i] + (1 - alpha * k[i]) * VIDYA[i-1]
    return VIDYA

#@feature_timeit
@jit(nopython=True)
def GMA(close: np.ndarray, period: int) -> np.ndarray[np.float64]:
    """Compute Geometric Moving Average using logarithms for efficiency."""
    gma = np.zeros(len(close))
    log_close = np.log(close)
    for i in range(period-1, len(close)):
        gma[i] = np.exp(np.mean(log_close[i-period+1:i+1]))
    return gma

#@feature_timeit
def FBA(close: np.ndarray, period: int) -> np.ndarray[np.float64]:
    fibs = []
    a,b = 0,1
    while b<=period:
       a,b = b,a+b
       fibs.append(b)
    #print(f'fibs {fibs}')
    moving_averages = []
    for i in fibs:
        # Use np.convolve to calculate the moving average
        moving_avg = np.convolve(close, np.ones(i), 'valid') / i
        # Append zeros at the beginning to match the original array's size
        moving_avg = np.concatenate((np.zeros(i-1), moving_avg))
        moving_averages.append(moving_avg)
    # Calculate the average of the moving averages
    #print(len(moving_averages))
    moving_averages = np.array(moving_averages)
    fma = np.mean(moving_averages, axis=0)
    return fma

#@feature_timeit
#@njit
'''def VAMA(close: np.ndarray, volume: np.ndarray, period: int) -> np.ndarray[np.float64]:
    volume_weights = close * volume
    volume_weights_sum = np.convolve(volume_weights.astype(np.float64), np.ones(period).astype(np.float64), mode='valid')
    volume_sum = np.convolve(volume, np.ones(period), 'valid')
    vama_values = volume_weights_sum / volume_sum
    return np.concatenate((np.full(period - 1, np.nan), vama_values))'''

#@feature_timeit
def anyMA_sig(np_close: np.ndarray, np_xMA: np.ndarray, np_ATR: np.ndarray, atr_multi: float=1.000) -> np.ndarray:
  #print(np_ATR)
  return ((np_xMA-np_close)/np_ATR)/atr_multi

######################################################################################
######################################################################################
######################################################################################

#@feature_timeit
def get_MA(np_df: np.ndarray, type: int, MA_period: int) -> np.ndarray:
   ma_types = {0: lambda np_df,period: RMA(np_df[:,3], timeperiod=period),
               1: lambda np_df,period: talib.SMA(np_df[:,3], timeperiod=period),
               2: lambda np_df,period: talib.EMA(np_df[:,3], timeperiod=period),
               3: lambda np_df,period: talib.WMA(np_df[:,3], timeperiod=period),
               4: lambda np_df,period: VWMA(np_df[:,3], np_df[:,4], timeperiod=period),
               5: lambda np_df,period: talib.KAMA(np_df[:,3], timeperiod=period),
               6: lambda np_df,period: talib.TRIMA(np_df[:,3], timeperiod=period),
               7: lambda np_df,period: talib.DEMA(np_df[:,3], timeperiod=period),
               8: lambda np_df,period: talib.TEMA(np_df[:,3], timeperiod=period),
               9: lambda np_df,period: talib.T3(np_df[:,3], timeperiod=period),
               10: lambda np_df,period: talib.MAMA(np_df[:,3])[0],
               11: lambda np_df,period: finTA.SMM(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               12: lambda np_df,period: finTA.SSMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               13: lambda np_df,period: finTA.VAMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               14: lambda np_df,period: finTA.ZLEMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), max(4,period)).to_numpy(),
               15: lambda np_df,period: finTA.EVWMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               16: lambda np_df,period: finTA.SMMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               17: lambda np_df,period: finTA.HMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), period).to_numpy(),
               18: lambda np_df,period: finTA.FRAMA(pd.DataFrame(np_df, columns=['open','high','low','close','volume','X']), 2*int(period//2), 2*int(period//2)//2).to_numpy(),
               19: lambda np_df,period: ti.ehma(np_df[:,3], period),
               20: lambda np_df,period: ti.lma(np_df[:,3], period),
               21: lambda np_df,period: ti.shmma(np_df[:,3], period),
               22: lambda np_df,period: ti.ahma(np_df[:,3], period),
               23: lambda np_df,period: ALMA(np_df[:,3], timeperiod=period),
               24: lambda np_df,period: HammingMA(np_df[:,3], period),
               25: lambda np_df,period: LSMA(np_df[:,3], max(3,period)),
               26: lambda np_df,period: LWMA(np_df[:,3],  period),
               27: lambda np_df,period: MGD(np_df[:,3], period),
               28: lambda np_df,period: GMA(np_df[:,3], period),
               29: lambda np_df,period: FBA(np_df[:,3], period),
               30: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=0),
               31: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=1),
               32: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=2),
               33: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=3),
               34: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=4),
               35: lambda np_df,period: NadarayWatsonMA(np_df[:,3], period, kernel=5)}
              #22: lambda np_df,period: VAMA(np_df[:,3], np_df[:,4], period),
              #31: lambda np_df,period: VIDYA(np_df[:,3], talib.CMO(np_df[:,3], period), period)
   return np.around(ma_types[type](np_df, MA_period),2)

def get_MA_signal(np_df: np.ndarray, type: int, MA_period: int, ATR_period: int, ATR_multi: float):
  #print(hex(id(np_df)))
  atr = talib.ATR(np_df[:,1], np_df[:,2], np_df[:,3], ATR_period)
  '''np_df[:,-1] = anyMA_sig(np_df[:,3],
                          get_MA(np_df, type, MA_period),
                          atr,
                          atr_multi=ATR_multi)'''
  #return np_df
  return anyMA_sig( np_df[:,3],
                    get_MA(np_df, type, MA_period),
                    atr,
                    atr_multi=ATR_multi )

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