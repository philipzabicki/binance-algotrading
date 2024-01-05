# binance-algotrading

Collection of personal tools used for strategies analysis and algorithmic trading on Binance exchange.

Main purpose of this repo is downloading historical data from exchanges, using them to create and evaluate trading
strategies and finally deplot it on server for live automatic trading on Binance exchange.

## Requirements
### Pre-requirements
Whole project runs with python 3.11.4, but it should work fine with any 3.11.x version.

Before installing TA-lib via pip you need to satisfy dependencies.
Just follow this [TA-Lib](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#dependencies).

Otherwise, you can download this file [ta_lib-0.4.25-cp311-cp311-win_amd64.whl](https://drive.google.com/file/d/117WDdPpTAJK_IX2yWpliBRy14m9uUWSD/view?usp=sharing).
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

## Download data
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
It imitates Binance Exchnage Spot market to some degree. Requires two dataframes to work, one with Dates and one with OHLCV values. One trading session is called episode and can use whole dataframe or randomly picked max_steps size data from df.

Allows to buy and sell an asset at any step using an 'action': {0 - hold, 1 - buy, 2 - sell}. One can also set stop loss for whole backtest period.

Always trades with current candle close price, allows to provide price slippage data for better imitation of real world scenario.

Backtesting works by calling 'step()' method with 'action' argument until max_steps is reached, episode ends or balance is so low it does not allow for any market action for given coin.
#### FuturesBacktest
It imitates Binance Exchange Futures market. Inherits from SpotBacktest. Requires additional dataframe with mark price ohlc values as binance uses mark prices for unrealized pnl calculation and liquidation price, see [this](https://www.binance.com/en/blog/futures/what-is-the-difference-between-a-futures-contracts-last-price-and-mark-price-5704082076024731087)

Adds new methods to allow [short selling](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L447), [margin checking](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L426), [postion tier checking](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin), [position liquidations](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L485) etc.
#### SignalExecuteSpotEnv and SignalExecuteFuturesEnv
Expands the SpotBacktest/FuturesBacktest class/environment to allow execution of single signal trading strategy all at once on whole episode (whole df or randomly picked max_steps size from whole dataframe).

Allows for asymmetrical enter position(enter_threshold) and close position(close_threshold) signals.

In generic base class implementation signals are empty numpy array. Other inheriting environments extend this.

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
### Technical analysis single signal trading environments
Inherited from SignalExecuteSpotEnv or SignalExecuteFuturesEnv.
Currently only MACD, MACD+RSI, Keltner Channel(Bands) and Chaikin Oscillator are implemented.

All of them can use [many custom](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/ta_tools.py#L1002) moving averages and any valid periods instead of just 12/26 periods EMAs.
Ex. MACD using TEMA for slow ma, HullMA for fast ma and HammingMA for signal line.
#### MACD strategy environment
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

## Trading strategies

...

## Launch live bot

...
