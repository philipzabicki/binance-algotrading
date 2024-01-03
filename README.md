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
```
def by_BinanceVision(ticker='BTCBUSD', interval='1m', market_type='um', data_type='klines', start_date='', split=False,
                     delay=LAST_DATA_POINT_DELAY):
  ...
```
and
```
def by_DataClient(ticker='BTCUSDT', interval='1m', futures=True, statements=True, split=False,
                  delay=LAST_DATA_POINT_DELAY):
  ...
```

## Backtesting enviroments

...

## Trading strategies

...

## Launch live bot

...
