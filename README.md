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
def by_DataClient(ticker='BTCUSDT',
                  interval='1m',
                  futures=True,
                  statements=True,
                  split=False,
                  delay=LAST_DATA_POINT_DELAY): ...
```
**Parameters:**

'split' and 'delay' are same as above. 
'interval' and 'statements' are ones from [binance_data](https://github.com/uneasyguy/binance_data#kline_data). Instead of 'pair_list' it uses only single ticker.
'futures' if false, downloads spot data.

## Backtesting enviroments

...

## Trading strategies

...

## Launch live bot

...
