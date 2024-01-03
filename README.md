# binance-algotrading

Collection of personal tools used for strategies analysis and algorithmic trading on Binance exchange.

Main purpose of this repo is downloading historical data from exchanges, using them to create and evaluate trading
strategies and finally deplot it on server for live automatic trading on Binance exchange.

## Requirements
### Pre-requirements
Whole project runs with python 3.11.4 but it should work fine with any 3.11.x

Before installing TA-lib via pip one needs to satisfy dependencies.
Just follow this [TA-Lib](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#dependencies).

Otherwise you can download this file [ta_lib-0.4.25-cp311-cp311-win_amd64.whl](https://drive.google.com/file/d/117WDdPpTAJK_IX2yWpliBRy14m9uUWSD/view?usp=sharing).
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
It will take some time...

## Download data
All data used in this project is either downloaded via [binance_data](https://github.com/uneasyguy/binance_data.git) packege or [Binance Vision url](https://data.binance.vision/)

Script used for handling data downloads is [utils/get_data.py](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/get_data.py)

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
split | BOOL | NO | If True splits Dates/Opened column from other columns (OHLCV usually) and function returns tuple (Opened_col, OHLCV_cols). Otherwise returns single df.
delay | INT | NO | Let's one decide data delay (in seconds) from most up-to-date datapoint. Uses constant value by default.

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

Allows for asymmetrical enter postion(enter_threshold) and close postion(close_threshold) signals.

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



## Trading strategies

...

## Launch live bot

...
