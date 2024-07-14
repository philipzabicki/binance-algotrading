from datetime import date
from io import BytesIO
from os import path, listdir
from time import time
from zipfile import ZipFile, BadZipFile

import pandas as pd
import requests
from binance_data import DataClient
from dateutil.relativedelta import relativedelta

from definitions import DATA_DIR

LAST_DATA_POINT_DELAY = 86_400  # 1 day in seconds
ITV_ALIASES = {'1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T'}


def _fix_and_fill_df(df, itv):
    # print(f'df before duplicates drop {df}')
    df.drop_duplicates(inplace=True)
    # print(f'df after duplicates drop {df}')
    ### We need to delete faulty rows with string values as DataClient seems to write it sometimes
    drop_values = ['Opened', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df[~df.isin(drop_values).any(axis=1)]
    ### data_range does use 'T' instead of 'm' to mark minute freq
    freq = itv if 'm' not in itv else ITV_ALIASES[itv]
    fixed_dates_df = pd.DataFrame(pd.date_range(start=df.iloc[0, 0], end=df.iloc[-1, 0], freq=freq), columns=['Opened'])
    # print(f'fixed_dates_df {fixed_dates_df}')
    df['Opened'] = pd.to_datetime(df['Opened'], format='%Y-%m-%d %H:%M:%S')
    if len(fixed_dates_df) > len(df):
        df = fixed_dates_df.merge(df, on='Opened', how='left')
        df.ffill(inplace=True)
    ### Replacing 0 volume with the smallest one to make some TAcalculations possible
    df.iloc[:, -1] = df.iloc[:, -1].replace(0.0, .00000001)
    return df


def _download_and_unzip(url, output_path):
    try:
        response = requests.get(url)
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_path)
        return True
    except BadZipFile:
        return False


def _read_partial_df(_path):
    file_path = path.join(_path, listdir(_path)[0])
    # print(f'file_path {file_path}')
    df_temp = pd.read_csv(file_path, sep=",", usecols=[0, 1, 2, 3, 4, 5])
    df_temp.columns = ['Opened', 'Open', 'High', 'Low', 'Close', 'Volume']  # Nadajemy nazwy kolumn
    df_temp['Opened'] = pd.to_datetime(df_temp['Opened'], unit='ms')
    # print(f'df_temp')
    # print(df_temp)
    return df_temp


def _collect_to_date(url, output_folder, start_date=date(year=2017, month=1, day=1), delta_itv='months'):
    if delta_itv == 'months':
        delta = relativedelta(months=1)
        end_date = date.today() - delta
        print(f'Collecting monthly from {start_date} to {end_date}')
    elif delta_itv == 'days':
        delta = relativedelta(days=1)
        end_date = date.today() - delta
        print(f'Collecting daily from {start_date} to {end_date}')
    else:
        raise ValueError("arg delta_itv should be one of 'months' or 'days'")
    data_frames = []
    while start_date <= end_date:
        if delta_itv == 'months':
            _url = url + f'{str(end_date)[:-3]}.zip'
            output_path = path.join(output_folder, f'{str(end_date)[:-3]}')
        elif delta_itv == 'days':
            _url = url + f'{end_date}.zip'
            output_path = path.join(output_folder, f'{end_date}')
        if path.exists(output_path) and listdir(output_path)[0].endswith('.csv'):
            data_frames.append(_read_partial_df(output_path))
        else:
            print(f'downloading... {_url}')
            if _download_and_unzip(_url, output_path):
                data_frames.append(_read_partial_df(output_path))
            else:
                print(f'"File is not a zip file" - {end_date} datapoint does not exist at BinanceVision')
                # break
        end_date -= delta
    # Collecting starts from current date and ends in last existing datapoint,
    # so we need to reverse df order
    data_frames.reverse()
    return data_frames


def by_BinanceVision(ticker='BTCBUSD',
                     interval='1m',
                     market_type='um',
                     data_type='klines',
                     start_date='',
                     end_date='',
                     split=False,
                     delay=LAST_DATA_POINT_DELAY):
    if end_date == "":
        end_date = date.today()
    # print("saved_args is", locals())
    ### example url #1 https://data.binance.vision/data/spot/daily/klines/BTCTUSD/1m/BTCTUSD-1m-2023-08-09.zip
    ### example url #2 https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2023-08-09.zip
    ### https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-07.zip
    ### https://data.binance.vision/data/spot/daily/klines/BTCFDUSD/1s/BTCFDUSD-1s-2023-09-23.zip
    if market_type == 'um' or market_type == 'cm':
        url = f'https://data.binance.vision/data/futures/{market_type}/monthly/{data_type}/{ticker}/{interval}/{ticker}-{interval}-'
        output_folder = DATA_DIR + f'binance_vision/futures_{market_type}/{data_type}/{ticker}{interval}'
    elif market_type == 'spot':
        url = f'https://data.binance.vision/data/{market_type}/monthly/{data_type}/{ticker}/{interval}/{ticker}-{interval}-'
        output_folder = DATA_DIR + f'binance_vision/{market_type}/{data_type}/{ticker}{interval}'
    else:
        raise ValueError("type argument must be one of { um, cm, spot }")
    print(f'Base url: {url}')
    print(f'Output folder: {output_folder}')

    if path.isfile(output_folder + '.csv'):
        df = pd.read_csv(output_folder + '.csv')
        df['Opened'] = pd.to_datetime(df['Opened'])
        last_timestamp = (df.iloc[-1]['Opened']).value // 10 ** 9
        print(f'time() - last_timestamp {time() - last_timestamp}')
        if (time() - last_timestamp) > delay:
            _start_date = pd.to_datetime(last_timestamp, unit='s').date()
            # print(f'start_date {start_date}')
            _end_date = date.today()
            data_frames = [df]
        else:
            if start_date != '':
                # print(fixed_df.loc[fixed_df['Opened'] >= start_date])
                df = df.loc[(df['Opened'] >= start_date) & (df['Opened'] <= end_date)]
            if split:
                return df.iloc[:, 0], df.iloc[:, 1:]
            else:
                return df
    # TODO: Fix problem when for the first few days of new month Binance Vision does not have aggregate monthly zip.
    else:
        data_frames = _collect_to_date(url, output_folder, delta_itv='months')
        _end_date = date.today()
        _start_date = date(_end_date.year, _end_date.month, 1)
    url = url.replace('monthly', 'daily')
    data_frames += _collect_to_date(url, output_folder, start_date=_start_date, delta_itv='days')
    df = pd.concat(data_frames, ignore_index=True)
    # print(df)
    fixed_df = _fix_and_fill_df(df, interval)
    fixed_df.to_csv(output_folder + '.csv', index=False)
    # print(fixed_df)
    if start_date != '':
        # print(fixed_df.loc[fixed_df['Opened'] >= start_date])
        fixed_df = fixed_df.loc[(fixed_df['Opened'] >= start_date) & (fixed_df['Opened'] <= end_date)]
    if split:
        return fixed_df.iloc[:, 0], fixed_df.iloc[:, 1:]
    else:
        return fixed_df


def by_DataClient(ticker='BTCUSDT',
                  interval='1m',
                  futures=True,
                  statements=True,
                  split=False,
                  delay=LAST_DATA_POINT_DELAY):
    directory = 'binance_data_futures/' if futures else 'binance_data_spot/'
    file = DATA_DIR + directory + interval + '_data/' + ticker + '/' + ticker + '.csv'
    ### example file path os.getcwd()+ /data/binance_data_spot/1m_data/BTCTUSD/BTCTUSD.csv
    if path.isfile(file):
        df = pd.read_csv(file, header=0)
        df['Opened'] = pd.to_datetime(df['Opened'])
        last_timestamp = (df.iloc[-1]['Opened']).value // 10 ** 9
        if time() - last_timestamp > delay:
            print(f'\n updating data... ({futures} {ticker} {interval} )')
            DataClient(futures=futures).kline_data([ticker.upper()], interval, storage=['csv', DATA_DIR + directory],
                                                   progress_statements=statements)
        else:
            return df
    else:
        print(f'\n downloading data... (futures={futures} {ticker} {interval})')
        DataClient(futures=futures).kline_data([ticker.upper()], interval, storage=['csv', DATA_DIR + directory],
                                               progress_statements=statements)
    df = pd.read_csv(file, header=0)
    fixed_df = _fix_and_fill_df(df, interval)
    if split:
        return fixed_df.iloc[:, 1], fixed_df.iloc[:, 1:]
    else:
        return fixed_df
