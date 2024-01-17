# binance-algotrading

This repository is a collection of personal tools designed for strategy analysis and algorithmic trading on the Binance exchange.

The main objective of this repository is to download historical data from exchange, utilize them for creating and evaluating trading strategies, and ultimately deploy them on a server for live automatic trading on the Binance exchange. I employ two approaches to develop trading strategies.

1. **Genetic Algorithm Approach:**
   Utilizes a genetic algorithm to optimize common technical analysis indicators and other trading variables (e.g., stop_loss). This approach aims to enhance the performance of trading strategies through evolutionary optimization.

2. **Reinforcement Learning Approach:**
   Applies reinforcement learning to create a profitable trading agent. This approach focuses on developing an intelligent agent that learns and adapts its trading strategies based on feedback from the market.

## Requirements
### Pre-requirements
Whole project runs with python 3.11.4, but it should work fine with any 3.11.x version.

Before installing TA-lib via pip you need to satisfy dependencies.
Just follow this [TA-Lib](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#dependencies).

Otherwise, you can download this file [ta_lib-0.4.25-cp311-cp311-win_amd64.whl](https://drive.google.com/file/d/117WDdPpTAJK_IX2yWpliBRy14m9uUWSD/view?usp=sharing)(TA-lib for python 3.11.4).
Then inside directory where you downloaded run 
```
pip install ta_lib-0.4.25-cp311-cp311-win_amd64.whl
```
### Install via pip
After satisfying TA-lib dependencies from above, you can download this repo as [zip](https://codeload.github.com/philipzabicki/binance-algotrading/zip/refs/heads/main) 

or by git CLI:
```
git clone https://github.com/philipzabicki/binance-algotrading.git
```
Then inside binance-algotrading directory run:
```
pip install -r requirements.txt
```
It may take some time...

## Getting data
All data used in this project is either downloaded via [binance_data](https://github.com/uneasyguy/binance_data.git) package or [Binance Vision](https://data.binance.vision/) website.
Website approach uses most efficient way to get data as it starts downloading from current date and goes back in time as long as there is data on website.
It also downloads data in monthly batches if available, if not daily are used.

Code used for handling data downloads can be found there: [utils/get_data.py](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/get_data.py)

Main functions in this file are:
### by_BinanceVision()
```python
def by_BinanceVision(ticker='BTCBUSD',
                     interval='1m',
                     market_type='um',
                     data_type='klines',
                     start_date='',
                     split=False,
                     delay=LAST_DATA_POINT_DELAY): ...
```
**Parameters:**
Name | Type | Mandatory | Description
------------ | ------------ | ------------ | ------------
ticker | STR | YES | Any cryptocurrency pair traded on Binance ex. 'ETHUSDT'
interval | STR | YES | Any trading interval existing on Binance Vision ex. '30m'
market_type | STR | YES | Options: 'um' - USDT-M Futures 'cm' - COIN-M Futures, 'spot' - Spot market
data_type | STR | YES | Futures options: 'aggTrades', 'bookDepth', 'bookTicker', 'indexPriceKlines', 'klines', 'liquidationSnapshot', 'markPriceKlines', 'metrics', 'premiumIndexKlines', 'trades'. Spot options: 'aggTrades', 'klines', 'trades'. Better explained with [Binance API](https://github.com/binance/binance-spot-api-docs/blob/master/web-socket-api.md#market-data-requests)
start_date | STR | NO | Any date format parsable by pandas datetime object. Best to use 'YYYY-MM-DD HH:MM:SS' or just 'YYYY-MM-DD'.
split | BOOL | NO | If True splits Dates/Opened column from other columns (OHLCV usually) and function returns tuple (Opened_col, OHLCV_cols). Otherwise, returns single df.
delay | INT | NO | Lets one decide data delay (in seconds) from most up-to-date datapoint. Uses constant value by default.

### by_DataClient()
```python
(...)
LAST_DATA_POINT_DELAY = 86_400  # 1 day in seconds
(...)

def by_DataClient(ticker='BTCUSDT',
                  interval='1m',
                  futures=True,
                  statements=True,
                  split=False,
                  delay=LAST_DATA_POINT_DELAY): (...)
```
**Parameters:**

'split' and 'delay' are same as for by_BinanceVision().

'interval' and 'statements' are ones from [binance_data](https://github.com/uneasyguy/binance_data#kline_data). Instead of 'pair_list' it handles only single ticker.

'futures' if false, downloads spot data.

## Backtesting environments
Project has 2 base [Gymnasium](https://github.com/Farama-Foundation/Gymnasium.git)/[Gym](https://github.com/openai/gym.git) compatible environments, [SpotBacktest](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L16) and [FuturesBacktest](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L352) (inheriting from SpotBacktest).

All other environments inherit from them.
### Base environments
#### SpotBacktest
It imitates Binance Exchnage Spot market to some degree. Requires two dataframes to work, one with Dates and one with OHLCV values. One trading session is called episode and can use whole dataframe or randomly picked max_steps sized data from df.

Allows to buy and sell an asset at any step using an 'action': {0 - hold, 1 - buy, 2 - sell}. One can also set stop loss for whole backtest period.

It always trades with current candle close price, you can provide price slippage data for better imitation of real world scenario.

Backtesting works by calling 'step()' method with 'action' argument until max_steps is reached, episode ends or balance is so low it does not allow for any market action for given coin.
#### FuturesBacktest
It imitates Binance Exchange Futures market. Inherits from SpotBacktest. Requires additional dataframe with mark price ohlc values as binance uses mark prices for unrealized pnl calculation and liquidation price, see [this](https://www.binance.com/en/blog/futures/what-is-the-difference-between-a-futures-contracts-last-price-and-mark-price-5704082076024731087)

Adds new methods to allow [short selling](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L447), [margin checking](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L426), [postion tier checking](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin), [position liquidations](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L485) etc.
#### SignalExecuteSpotEnv and SignalExecuteFuturesEnv
Expands the SpotBacktest/FuturesBacktest class/environment to allow execution of single signal trading strategy all at once on whole episode (whole df or randomly picked max_steps size from whole dataframe).

Allows for asymmetrical enter position(enter_threshold) and close position(close_threshold) signals.

In generic base class implementation signals are empty numpy array. Other inheriting environments extend this.

SignalExecute-like object when called, executes whole trading episode on given signals array. Negative values are reserved for short/sell singals. Positive for long/buy.

##### SignalExecuteSpotEnv
```python
(...)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'enter_at' in kwargs:
            self.enter_threshold = kwargs['enter_at']
            self.close_threshold = kwargs['close_at']
        else:
            self.enter_threshold = 1.0
            self.close_threshold = 1.0
        self.signals = empty(self.total_steps)

(...)

    def __call__(self, *args, **kwargs):
        while not self.done:
            # step must be start_step adjusted cause one can start and end backtest at any point in df
            _step = self.current_step - self.start_step
            if self.signals[_step] >= self.enter_threshold:
                action = 1
            elif self.signals[_step] <= -self.close_threshold:
                action = 2
            else:
                action = 0
            self.step(action)
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render(indicator_or_reward=self.signals[_step])
                self.current_step += 1
        return None, self.reward, self.done, False, self.info
```

##### SignalExecuteFuturesEnv
```python
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'long_enter_at' in kwargs and 'short_close_at' in kwargs:
            self.long_enter_threshold = kwargs['long_enter_at']
            self.long_close_threshold = kwargs['long_close_at']
            self.short_enter_threshold = kwargs['short_enter_at']
            self.short_close_threshold = kwargs['short_close_at']
            self.leverage = kwargs['leverage']
        else:
            self.long_enter_threshold = 1.0
            self.long_close_threshold = 1.0
            self.short_enter_threshold = 1.0
            self.short_close_threshold = 1.0
            self.leverage = 5
        self.signals = empty(self.total_steps)

   (...)

    def __call__(self, *args, **kwargs):
     while not self.done:
         # step must be start_step adjusted cause one can start and end backtest at any point in df
         _step = self.current_step - self.start_step
         if self.qty == 0 and self.signals[_step] >= self.long_enter_threshold:
             action = 1
         elif self.qty == 0 and self.signals[_step] <= -self.short_enter_threshold:
             action = 2
         elif self.qty > 0 and self.signals[_step] <= -self.long_close_threshold:
             action = 2
         elif self.qty < 0 and self.signals[_step] >= self.short_close_threshold:
             action = 1
         else:
             action = 0
         self.step(action)
         if self.visualize:
             # current_step manipulation just to synchronize plot rendering
             # could be fixed by calling .render() inside .step() just before return statement
             self.current_step -= 1
             self.render(indicator_or_reward=self.signals[_step])
             self.current_step += 1
     return None, self.reward, self.done, False, self.info
```

### Technical analysis single signal trading environments
Inherited from SignalExecuteSpotEnv or SignalExecuteFuturesEnv.
Currently only MACD, MACD+RSI, Keltner Channel(Bands) and Chaikin Oscillator are implemented.

All of MA based indicators can use [many custom](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/ta_tools.py#L1002) moving averages with any valid periods.
Ex. MACD using TEMA-42 for slow ma, HullMA-37 for fast ma and HammingMA-8 for signal line.

The MAs used for optimizations are listed in function inside utils/ta_tools.py. Ones used now are the fast enough ones, some new may appear in future.
```python
def get_MA(np_df: np.ndarray, ma_type: int, ma_period: int) -> np.ndarray:
    """
        Calculate Moving Average (MA) based on the specified MA type and period.
        Parameters:
        - np_df (np.ndarray): Numpy array containing OHLCV (Open, High, Low, Close, Volume) data.
        - ma_type (int): Type of Moving Average to calculate. Choose from the following options:
            0: Simple Moving Average (SMA)
            1: Exponential Moving Average (EMA)
            2: Weighted Moving Average (WMA)
            3: Kaufman's Adaptive Moving Average (KAMA)
            4: Triangular Moving Average (TRIMA)
            5: Double Exponential Moving Average (DEMA)
            6: Triple Exponential Moving Average (TEMA)
            7: Triple Exponential Moving Average (T3)
            8: MESA Adaptive Moving Average (MAMA)
            9: Linear Regression Moving Average (LINEARREG)
            10: Simple Moving Median (SMM)
            11: Smoothed Simple Moving Average (SSMA)
            12: Volume Adjusted Moving Average (VAMA)
            13: Zero Lag Exponential Moving Average (ZLEMA)
            14: Exponential Volume Weighted Moving Average (EVWMA)
            15: Smoothed Moving Average (SMMA)
            16: Volume Weighted Moving Average (VWMA)
            17: Symmetrically Weighted Moving Average (SWMA) - Ascending
            18: Symmetrically Weighted Moving Average (SWMA) - Descending
            19: Exponential Hull Moving Average (EHMA)
            20: Leo Moving Average (LMA)
            21: Sharp Modified Moving Average (SHMMA)
            22: Ahrens Moving Average (AHMA)
            23: Hull Moving Average (HullMA)
            24: Volume Weighted Moving Average (VWMA)
            25: Relative Moving Average (RMA)
            26: Arnaud Legoux Moving Average (ALMA)
            27: Hamming Moving Average (HammingMA)
            28: Linear Weighted Moving Average (LWMA)
            29: McGinley Dynamic (MGD)
            30: Geometric Moving Average (GMA)
            31: Fibonacci Based Average (FBA)
            32: Nadaray-Watson Moving Average (kernel 0 - gaussian)
            33: Nadaray-Watson Moving Average (kernel 1 - epanechnikov)
            34: Nadaray-Watson Moving Average (kernel 2 - rectangular)
            35: Nadaray-Watson Moving Average (kernel 3 - triangular)
            36: Nadaray-Watson Moving Average (kernel 4 - biweight)
            37: Nadaray-Watson Moving Average (kernel 5 - cosine)
        - ma_period (int): Number of periods for the Moving Average calculation.
        Returns:
        - np.ndarray: Numpy array containing the calculated Moving Average values.
        """
    ma_types = {0: lambda ohlcv_array, period: talib.SMA(ohlcv_array[:, 3], timeperiod=period),
                1: lambda ohlcv_array, period: talib.EMA(ohlcv_array[:, 3], timeperiod=period),
                2: lambda ohlcv_array, period: talib.WMA(ohlcv_array[:, 3], timeperiod=period),
                3: lambda ohlcv_array, period: talib.KAMA(ohlcv_array[:, 3], timeperiod=period),
                4: lambda ohlcv_array, period: talib.TRIMA(ohlcv_array[:, 3], timeperiod=period),
                5: lambda ohlcv_array, period: talib.DEMA(ohlcv_array[:, 3], timeperiod=period),
                6: lambda ohlcv_array, period: talib.TEMA(ohlcv_array[:, 3], timeperiod=period),
                7: lambda ohlcv_array, period: talib.T3(ohlcv_array[:, 3], timeperiod=period),
                8: lambda ohlcv_array, period: talib.MAMA(ohlcv_array[:, 3])[0],
                9: lambda ohlcv_array, period: talib.LINEARREG(ohlcv_array[:, 3], timeperiod=period),
                10: lambda ohlcv_array, period: finTA.SMM(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    period).to_numpy(),
                11: lambda ohlcv_array, period: finTA.SSMA(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    period).to_numpy(),
                12: lambda ohlcv_array, period: finTA.VAMA(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    period).to_numpy(),
                13: lambda ohlcv_array, period: finTA.ZLEMA(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    max(4, period)).to_numpy(),
                14: lambda ohlcv_array, period: finTA.EVWMA(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    period).to_numpy(),
                15: lambda ohlcv_array, period: finTA.SMMA(
                    pd.DataFrame(ohlcv_array[:, :5], columns=['open', 'high', 'low', 'close', 'volume']),
                    period).to_numpy(),
                16: lambda ohlcv_array, period: p_ta.vwma(pd.Series(ohlcv_array[:, 3]),
                                                          pd.Series(ohlcv_array[:, 4]),
                                                          length=period).to_numpy(),
                17: lambda ohlcv_array, period: p_ta.swma(pd.Series(ohlcv_array[:, 3]),
                                                          length=period,
                                                          asc=True).to_numpy(),
                18: lambda ohlcv_array, period: p_ta.swma(pd.Series(ohlcv_array[:, 3]),
                                                          length=period,
                                                          asc=False).to_numpy(),
                19: lambda ohlcv_array, period: ti.ehma(ohlcv_array[:, 3], period),
                20: lambda ohlcv_array, period: ti.lma(ohlcv_array[:, 3], period),
                21: lambda ohlcv_array, period: ti.shmma(ohlcv_array[:, 3], period),
                22: lambda ohlcv_array, period: ti.ahma(ohlcv_array[:, 3], period),
                23: lambda ohlcv_array, period: HullMA(ohlcv_array[:, 3], max(period, 4)),
                24: lambda ohlcv_array, period: VWMA(ohlcv_array[:, 3], ohlcv_array[:, 4], timeperiod=period),
                25: lambda ohlcv_array, period: RMA(ohlcv_array[:, 3], timeperiod=period),
                26: lambda ohlcv_array, period: ALMA(ohlcv_array[:, 3], timeperiod=period),
                27: lambda ohlcv_array, period: HammingMA(ohlcv_array[:, 3], period),
                28: lambda ohlcv_array, period: LWMA(ohlcv_array[:, 3], period),
                29: lambda ohlcv_array, period: MGD(ohlcv_array[:, 3], period),
                30: lambda ohlcv_array, period: GMA(ohlcv_array[:, 3], period),
                31: lambda ohlcv_array, period: FBA(ohlcv_array[:, 3], period),
                32: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=0),
                33: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=1),
                34: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=2),
                35: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=3),
                36: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=4),
                37: lambda ohlcv_array, period: NadarayWatsonMA(ohlcv_array[:, 3], period, kernel=5)}
    return ma_types[ma_type](np_df.astype(np.float64), ma_period)
```
#### MACD
Expands SignalExecuteSpotEnv/SignalExecuteFuturesEnv by creating signal array from MACD made with arguments provided to reset method.
##### Execute environment
```python
(...)
class MACDExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              fast_period=12, slow_period=26, signal_period=9,
              fast_ma_type=0, slow_ma_type=0, signal_ma_type=0, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, enter_at=enter_at, close_at=close_at, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        self.signal_ma_type = signal_ma_type
        _max_period = max(self.fast_period, self.slow_period) + self.signal_period
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        # print(self.df[self.start_step:self.end_step, :5])
        macd, macd_signal = custom_MACD(self.df[prev_values:self.end_step, :5],
                                        fast_ma_type=fast_ma_type, fast_period=fast_period,
                                        slow_ma_type=slow_ma_type, slow_period=slow_period,
                                        signal_ma_type=signal_ma_type, signal_period=signal_period)
        self.signals = MACD_cross_signal(macd[self.start_step - prev_values:],
                                         macd_signal[self.start_step - prev_values:])
        return _ret
```
Custom MACD is calculated using [known formula](https://www.investopedia.com/terms/m/macd.asp) implemented as function inside [utils/ta_tools.pl](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/ta_tools.py#L1131)

What's less known is way I derive signals from this indicator:
```python
@jit(nopython=True, nogil=True, cache=True)
def MACD_cross_signal(macd_col: list | np.ndarray, signal_col: list | np.ndarray) -> list[float | int]:
    return [0.0] + [1 if (cur_sig < 0) and (cur_macd < 0) and (cur_macd > cur_sig) and (prev_macd < prev_sig) else
                    .75 if (cur_macd > cur_sig) and (prev_macd < prev_sig) else
                    -1 if (cur_sig > 0) and (cur_macd > 0) and cur_macd < cur_sig and prev_macd > prev_sig else
                    -.75 if cur_macd < cur_sig and prev_macd > prev_sig else
                    0.5 if cur_macd > prev_macd and cur_sig < prev_sig else
                    -0.5 if cur_macd < prev_macd and cur_sig > prev_sig else
                    0
                    for cur_sig, cur_macd, prev_sig, prev_macd in
                    zip(signal_col[1:], macd_col[1:], signal_col[:-1], macd_col[:-1])]
```
As mentioned earlier, negative values indicate short/sell singals, positive - long/buy.
Crossing logic is as usual, signal line crosses macd from above - short, from below - long.

As you can see the highest signal values are 1 and -1, which are generated when lines cross above(for short) or below(for long) 0 level.

Slightly weaker singlas 0.75 and -0.75 are generated when lines cross but without additional above or below zero level logic.

0.5 and -0.5 signal values are generated when lines are getting closer to each other(approaching crossing).
##### Strategy environment
#### Keltner Channel(Bands)
##### Execute environment
```python
class BandsExecuteSpotEnv(SignalExecuteSpotEnv):
    def reset(self, *args, stop_loss=None, enter_at=1.0, close_at=1.0,
              atr_multi=1.0, atr_period=1,
              ma_type=0, ma_period=1, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, enter_at=enter_at, close_at=close_at, **kwargs)
        self.ma_type = ma_type
        self.ma_period = ma_period
        self.atr_period = atr_period
        self.atr_multi = atr_multi
        _max_period = max(self.ma_period, self.atr_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        self.signals = get_MA_band_signal(self.df[prev_values:self.end_step, :5],
                                          self.ma_type, self.ma_period,
                                          self.atr_period, self.atr_multi)[self.start_step - prev_values:]
        return _ret
```
##### Strategy environment
#### Chaikin Oscillator
##### Execute environment
```python
class ChaikinOscillatorExecuteSpotEnv(SignalExecuteSpotEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adl = AD(self.df[:, 1], self.df[:, 2], self.df[:, 3], self.df[:, 4])

    def reset(self, *args, stop_loss=None,
              fast_period=3, slow_period=10,
              fast_ma_type=0, slow_ma_type=0, **kwargs):
        _ret = super().reset(*args, stop_loss=stop_loss, **kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.fast_ma_type = fast_ma_type
        self.slow_ma_type = slow_ma_type
        _max_period = max(self.fast_period, self.slow_period)
        if _max_period > self.total_steps:
            raise ValueError('One of indicator periods is greater than df size.')
        # Calculate only the data length necessary, with additional length caused by indicator periods
        prev_values = self.start_step - _max_period if self.start_step > _max_period else 0
        chaikin_oscillator = custom_ChaikinOscillator(self.adl[prev_values:self.end_step, ],
                                                      fast_ma_type=fast_ma_type, fast_period=fast_period,
                                                      slow_ma_type=slow_ma_type, slow_period=slow_period)
        self.signals = ChaikinOscillator_signal(chaikin_oscillator[self.start_step - prev_values:])
        return _ret
```
##### Strategy environment
#### Any other new strategy environment
You can create any other TA indicator trading environment, using ones already existing as template, just take care that your generated signals are properly calculated.
## Genetic Algorithm trading strategies
One way to finding optimal technical analysis parameters for trading is to use Genetic Algorithm. All previously showed strategy environments can be used for optimization.
### Environments
#### Base
#### Env1
#### Env2
#### Env3
## Reinforcement Learning trading agent
<DESC>
### RL Environment
...

## Live trading bot

...
