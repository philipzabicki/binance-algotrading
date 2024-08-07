# binance-algotrading [README in progress...]

This repository is a collection of personal tools designed for strategy analysis and algorithmic trading on the Binance
exchange. The whole project is written entirely by me in my free time.

The main objective of this repository is to get historical data from exchange, utilize them for creating and
evaluating trading strategies, and ultimately deploy those strategies for live automatic trading on the exchange.

I employ technical analysis approaches to develop trading strategies.

**Genetic Algorithm Parametrization:**
   Utilizes a genetic algorithm to optimize common technical analysis indicators and other trading variables (e.g.,
   stop_loss). This approach aims to enhance the performance of trading strategies through evolutionary optimization.

## Requirements

### Pre-requirements

Whole project runs with python 3.11.9, but it should work fine with any 3.11.x version.

Before installing TA-lib via pip you need to satisfy dependencies.
Just follow this [TA-Lib](https://github.com/TA-Lib/ta-lib-python?tab=readme-ov-file#dependencies).

Otherwise, you can download this
file [ta_lib-0.4.25-cp311-cp311-win_amd64.whl](https://drive.google.com/file/d/117WDdPpTAJK_IX2yWpliBRy14m9uUWSD/view?usp=sharing)(
TA-lib for python 3.11.x).
Then inside directory where you downloaded run

```
pip install ta_lib-0.4.25-cp311-cp311-win_amd64.whl
```

### Install via pip

After satisfying TA-lib dependencies from above, you can download this repo
as [zip](https://codeload.github.com/philipzabicki/binance-algotrading/zip/refs/heads/main)

or by git CLI:

```
git clone https://github.com/philipzabicki/binance-algotrading.git
```

Then inside binance-algotrading directory run:

```
pip install -r requirements.txt
```

It is important that you download my patch for finta package (included in requirements.txt)
as It needed a small adjustment to work with python 3.11.x

## Getting data

All data used in this project is either downloaded via [binance_data](https://github.com/uneasyguy/binance_data.git)
package or [Binance Vision](https://data.binance.vision/) website.
Website approach uses the most efficient way
to get data as it starts downloading from current date and goes back in time as
long as there is data on website.
It also downloads data in monthly batches if available otherwise not daily is used.

Code used for handling data downloads can be found
there: [utils/get_data.py](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/get_data.py)

Main functions in this file are:

### by_BinanceVision()

```python
LAST_DATA_POINT_DELAY = 86_400  # 1 day in seconds

def by_BinanceVision(ticker='BTCBUSD',
                     interval='1m',
                     market_type='um',
                     data_type='klines',
                     start_date='',
                     end_date='',
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
end_date | STR | NO | Same as above.
split | BOOL | NO | If True splits Dates/Opened column from other columns (OHLCV usually) and function returns tuple (Opened_col, OHLCV_cols). Otherwise, returns single df.
delay | INT | NO | Lets one decide data delay (in seconds) from most up-to-date datapoint. Uses constant value by default.

### by_DataClient()

```python
LAST_DATA_POINT_DELAY = 86_400  # 1 day in seconds

def by_DataClient(ticker='BTCUSDT',
                  interval='1m',
                  futures=True,
                  statements=True,
                  split=False,
                  delay=LAST_DATA_POINT_DELAY): ...
```

**Parameters:**

'split' and 'delay' are same as for by_BinanceVision().

'interval' and 'statements' are ones from [binance_data](https://github.com/uneasyguy/binance_data#kline_data). Instead
of 'pair_list' it handles only single ticker.

'futures' if false, downloads spot data.

## Backtesting environments

Project has 2
base [Gymnasium](https://github.com/Farama-Foundation/Gymnasium.git)/[Gym](https://github.com/openai/gym.git) compatible
environments, [SpotBacktest](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L16)
and [FuturesBacktest](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L352) (inheriting from SpotBacktest).

All other environments inherit from them.

### Base environments

#### SpotBacktest

It imitates Binance Exchnage Spot market to some degree. Requires dataframe with OHLCV values for wanted ticker to work.
Runs by calling step method with trading action. One full trading run is called episode. After every episode, you need to reset the environment with reset method.  

```python
class SpotBacktest(Env):
    def __init__(self, df, start_date='', end_date='', max_steps=0, exclude_cols_left=1, no_action_finish=2_880,
                 init_balance=1_000, position_ratio=1.0, save_ratio=None, stop_loss=None, take_profit=None,
                 fee=0.0002, coin_step=0.001, slippage=None, slipp_std=0,
                 render_range=120, verbose=True, visualize=False, write_to_file=False, *args, **kwargs): ...
```

**Parameters:**
Name | Type | Mandatory | Description
------------ | ------------ | ------------ | ------------
df | pandas.DataFrame | YES | A dataframe with OHLCV values for any ticker/coin pair
start_date | STR | NO | Env will only run episodes starting from this date in df, format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
end_date | STR | NO | Env will only run episodes ending at this date in df, format: 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
max_steps | INT | NO | Max steps played in single episode, if larger than 0, location inside df is randomly picked.
exclude_cols_left | INT | NO | Index of columns to be excluded from observation space (df) e.g. value 1 will skip first column (counting from left) from df that means 'Opened' (dates col) will not show in observation space
no_action_finish | INT | YES | When no trading action is taken for this many steps, episode will be forced to end 
init_balance | INT/FLOAT | YES | Initial balance for trading episode to start with (in quote coin)
position_ratio | FLOAT | YES | Values from 0.0 to 1.0, E.g. 0.5 means only 50% of available balance will be used for trading
save_ratio | FLOAT | NO | This part of absolute profit from any profitable trade will be saved to save_balance (which is not used for trading)
stop_loss | FLOAT | NO | This stop loss will be used for all episodes played, it is applied to enter position close price value not position PnL
take_profit | FLOAT | NO | Similarly to above
fee | FLOAT | YES | Fee paid for every buy and sell action
coin_step | FLOAT | YES | Minimal trade amount for ticker base coin, use values from [spot trading rules](https://www.binance.com/en/trade-rule)
slippage | DICTIONARY | NO | Python dict with price slippage data. First value from tuple for any key is mean slippage and second is standard deviation e.g. {'buy': (1.0, 0.0), 'sell': (1.0, 0.0)}
slipp_std | INT | NO | This many standard deviations from mean (slippage dict) will be taken to calculate real buy/sell prices
visualize | BOOL | YES | If true, every step of episode will be rendered to screen with chart visualization
render_range | INT | NO | Only applies when visualize=True, this many steps will be rendered in visualization
verbose | BOOL | YES | If true, at end of every episode env will print summary statistics to console
write_to_file | BOOL | YES | If true, env will save state to csv every step with buy/sell action


Trading actions possible are: {0 - hold, 1 - buy, 2 - sell}. It always trades with current candle close price, you can provide price slippage data for better imitation of a real world
scenario.
You can also set stop loss and take profit for a whole backtest episode.
Setting a save_ratio parameter forces env to save some part of trading profit and do not use it for future trading.

Backtesting works by calling 'step()' method with 'action' argument until max_steps is reached, df ends or balance
is so low it does not allow for any market action for given coin.
    
#### FuturesBacktest

It imitates Binance Exchange Futures market. Inherits from SpotBacktest. Requires additional dataframe with mark price
ohlc values as binance uses mark prices for unrealized pnl calculation and liquidation price,
see [this](https://www.binance.com/en/blog/futures/what-is-the-difference-between-a-futures-contracts-last-price-and-mark-price-5704082076024731087). It also adds new constructor parameter - leverage. For now works well only for BTCUSDT perpetual as
it requires adding variable (by ticker) leverage range and position tier updates. [BTCUSDT.P tiers](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin)

Expands SpotBacktest env class with new methods to
allow [short selling](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L447), [margin checking](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L426), [postion tier checking](https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin), [position liquidations](https://github.com/philipzabicki/binance-algotrading/blob/main/enviroments/base.py#L485)
etc.

### Signal environments
#### SignalExecuteSpotEnv and SignalExecuteFuturesEnv

Expands the SpotBacktest/FuturesBacktest class/environment to allow execution of trading strategy at
once on full episode accordingly to given trading signals array.

Allows for asymmetrical enter position(enter_threshold) and close position(close_threshold) signals. When using Futures version
the asymmetrical aspect is applied for long and short separately (4 thresholds). 

In generic class implementation signals are random numpy array. Other inheriting environments change this by creating trading signals using technical analysis. 

SignalExecute-like object when called, executes trading episode with signals array using position enter/close
thresholds values to determine position side for all trades. Negative values are reserved for short/sell signals.
Positive for long/buy.

For e.g.:

* signals = [0.52, 0, 0, -0.78]
* enter_threshold = 0.5
* close_threshold = 0.75

will result in a trading sequence of BUY -> HOLD -> HOLD -> SELL actions.

##### SignalExecuteSpotEnv

```python
from numpy.random import choice

class SignalExecuteSpotEnv(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.0
        self.save_ratio = kwargs['save_ratio'] if 'save_ratio' in kwargs else None
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
        self.enter_threshold = kwargs['enter_at'] if 'enter_at' in kwargs else 1.0
        self.close_threshold = kwargs['close_at'] if 'close_at' in kwargs else 1.0
        # self.signals = empty(self.total_steps)
        self.signals = choice([-1, 0, 1], size=self.total_steps)

    def reset(self, *args, **kwargs):
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.0
        self.save_ratio = kwargs['save_ratio'] if 'save_ratio' in kwargs else None
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
        self.enter_threshold = kwargs['enter_at'] if 'enter_at' in kwargs else 1.0
        self.close_threshold = kwargs['close_at'] if 'close_at' in kwargs else 1.0
        return super().reset()

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
It is similar to above but expands enter and close threshold with long and short possibilities.

```python
from numpy.random import choice

class SignalExecuteFuturesEnv(FuturesBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.0
        self.save_ratio = kwargs['save_ratio'] if 'save_ratio' in kwargs else None
        self.leverage = kwargs['leverage'] if 'leverage' in kwargs else 1
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
        self.long_enter_threshold = kwargs['long_enter_at'] if 'long_enter_at' in kwargs else 1.0
        self.long_close_threshold = kwargs['long_close_at'] if 'long_close_at' in kwargs else 1.0
        self.short_enter_threshold = kwargs['short_enter_at'] if 'short_enter_at' in kwargs else 1.0
        self.short_close_threshold = kwargs['short_close_at'] if 'short_close_at' in kwargs else 1.0
        self.signals = choice([-1, 0, 1], size=self.total_steps)

    def reset(self, *args, **kwargs):
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.0
        self.save_ratio = kwargs['save_ratio'] if 'save_ratio' in kwargs else None
        self.leverage = kwargs['leverage'] if 'leverage' in kwargs else 1
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.take_profit = kwargs['take_profit'] if 'take_profit' in kwargs else None
        self.long_enter_threshold = kwargs['long_enter_at'] if 'long_enter_at' in kwargs else 1.0
        self.long_close_threshold = kwargs['long_close_at'] if 'long_close_at' in kwargs else 1.0
        self.short_enter_threshold = kwargs['short_enter_at'] if 'short_enter_at' in kwargs else 1.0
        self.short_close_threshold = kwargs['short_close_at'] if 'short_close_at' in kwargs else 1.0
        return super().reset()


    def __call__(self, *args, **kwargs):
        while not self.done:
            # step must be start_step adjusted cause one can start and end backtest at any point in df
            _step = self.current_step - self.start_step
            action = 0
            if self.qty == 0:
                if self.signals[_step] >= self.long_enter_threshold:
                    action = 1
                elif self.signals[_step] <= -self.short_enter_threshold:
                    action = 2
            elif self.qty > 0 and self.signals[_step] <= -self.long_close_threshold:
                action = 2
            elif self.qty < 0 and self.signals[_step] >= self.short_close_threshold:
                action = 1
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

Those environments inherit from SignalExecuteSpotEnv or SignalExecuteFuturesEnv 
and work by creating trading signals array for given technical analysis indicator and its settings.
Currently only MACD, Keltner Channel(Bands) and Chaikin Oscillator are implemented.

#### MACD example

Expands SignalExecuteSpotEnv/SignalExecuteFuturesEnv by creating signal array from MACD indicator created with arguments provided to
reset method.

##### Execute environment
These environments overload the reset method of the base class by adding to it the calculation of an array of signals resulting from the MACD indicator.
The MACD is calculated based on the parameters passed as an argument.

```python
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

Custom MACD is calculated using [known formula](https://www.investopedia.com/terms/m/macd.asp) implemented as function
inside [utils/ta_tools.pl](https://github.com/philipzabicki/binance-algotrading/blob/main/utils/ta_tools.py#L1131)

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

As you can see the highest signal values are 1 and -1, which are generated when lines cross above(for short) or below(
for long) 0 level.

Slightly weaker singlas 0.75 and -0.75 are generated when lines cross but without additional above or below zero level
logic.

0.5 and -0.5 signal values are generated when lines are getting closer to each other(approaching crossing).

#### Any other new technical analysis signal trading environment

I will not describe the other environments, as they work on the same principle as the environment for MACD described earlier.
You can create any other TA indicator trading environment, using ones already existing as template, just take care that
your generated signals are properly calculated.

## Genetic algorithm for parameters optimization

One way to finding optimal technical analysis parameters for trading is to use genetic algorithms. All previously showed
strategy environments can be used for optimization.

### Optimization environments
Like the signal environments described earlier, these go a step further and use these environments internally
to execute trading episodes and by calling the reset method with the given parameters the internal signal environments
are also reset and create a new signal array based on the given parameters.

#### Keltner Channel(Bands) example
I will describe the principle of these environments using the Bands ones as example.
This environment has an additional saving parameter that indicates what percentage of the profits from each trade
is to be retained and not used in subsequent trades. 

```python
class BandsOptimizeSavingFuturesEnv(Env):
    def __init__(self, *args, **kwargs):
        self.exec_env = _BandsExecuteFuturesEnv(*args, **kwargs)
        obs_lower_bounds = array([-inf for _ in range(8)])
        obs_upper_bounds = array([inf for _ in range(8)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)
        ### ACTION BOUNDARIES ###
        action_lower = [0.01, 0.000, 0.0001, 0.0001, 0.001, 0.001, 0.001, 0.001, 0.001, 2, 0, 2, 1]
        action_upper = [1.00, 1.000, 0.0500, 1.0000, 1.000, 1.000, 1.000, 1.000, 15.0, 10_000, 37, 10_000, 125]
        #########################
        self.action_space = spaces.Box(low=array(action_lower), high=array(action_upper), dtype=float64)

    def reset(self, position_ratio=1.0, stop_loss=None, take_profit=None, save_ratio=None,
              long_enter_at=1.0, long_close_at=1.0,
              short_enter_at=1.0, short_close_at=1.0, leverage=1,
              ma_type=0, ma_period=2,
              atr_period=2, atr_multi=1.0):
        return self.exec_env.reset(position_ratio=position_ratio, save_ratio=save_ratio,
                                   stop_loss=stop_loss, take_profit=take_profit,
                                   long_enter_at=long_enter_at, long_close_at=long_close_at,
                                   short_enter_at=short_enter_at, short_close_at=short_close_at,
                                   atr_multi=atr_multi, atr_period=atr_period,
                                   ma_type=ma_type, ma_period=ma_period,
                                   leverage=leverage)

    def step(self, action):
        self.reset(position_ratio=action[0], save_ratio=action[1],
                   stop_loss=action[2], take_profit=action[3],
                   long_enter_at=action[4], long_close_at=action[5],
                   short_enter_at=action[6], short_close_at=action[7],
                   atr_multi=action[8], atr_period=int(action[9]),
                   ma_type=int(action[10]), ma_period=int(action[11]), leverage=int(action[12]))
        return self.exec_env()
```

### Parametrizer environments

#### Envx

## Live trading bots
