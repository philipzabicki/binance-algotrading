# import pandas as pd
from os import path, listdir
from definitions import DATA_DIR
import requests
from io import BytesIO
from zipfile import ZipFile
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
from binance_data import DataClient
from time import time

LAST_DATA_POINT_DELAY = 86_400  # in seconds
ITV_ALIASES = {'1m': '1T', '3m': '3T', '5m': '5T', '15m': '15T', '30m': '30T'}


def fix_and_fill_df(df, itv):
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
    df = fixed_dates_df.merge(df, on='Opened', how='left')
    df.ffill(inplace=True)
    ### Replacing 0 volume with really small one to make some TAcalculations possible
    df.iloc[:, -1] = df.iloc[:, -1].replace(0.0, .00000001)
    return df


def download_and_unzip(url, output_path):
    try:
        response = requests.get(url)
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(output_path)
        return True
    except Exception as e:
        print(e)
        return False


def read_partial_df(_path):
    file_path = path.join(_path, listdir(_path)[0])
    # print(f'file_path {file_path}')
    df_temp = pd.read_csv(file_path, sep=",", usecols=[0, 1, 2, 3, 4, 5])
    df_temp.columns = ['Opened', 'Open', 'High', 'Low', 'Close', 'Volume']  # Nadajemy nazwy kolumn
    df_temp['Opened'] = pd.to_datetime(df_temp['Opened'], unit='ms')
    # print(f'df_temp')
    # print(df_temp)
    return df_temp


def collect_monthly(url, output_folder):
    # print(f'output_folder {output_folder}')
    end_date = date.today() - relativedelta(months=1)
    # print(f'end_date {end_date}')
    start_date = date(year=2017, month=1, day=1)
    # print(f'start_date {start_date}')
    data_frames = []
    while start_date <= end_date:
        _url = url + f'{str(start_date)[:-3]}.zip'
        output_path = path.join(output_folder, f'{str(start_date)[:-3]}')
        # print(f'output_path {output_path}')
        if path.exists(output_path) and listdir(output_path)[0].endswith('.csv'):
            # print(f'os.listdir(output_path)[0] {os.listdir(output_path)[0]}')
            data_frames.append(read_partial_df(output_path))
        else:
            print(f'downloading... {_url}')
            if download_and_unzip(_url, output_path):
                data_frames.append(read_partial_df(output_path))
            else:
                None
                # print('File does not exist.')
        start_date += relativedelta(months=1)
        # print(start_date)
    return data_frames


def by_BinanceVision(ticker='BTCBUSD', interval='1m', type='um', data='klines', delay=LAST_DATA_POINT_DELAY):
    ### example url #1 https://data.binance.vision/data/spot/daily/klines/BTCTUSD/1m/BTCTUSD-1m-2023-08-09.zip
    ### example url #2 https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2023-08-09.zip
    ### https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-07.zip
    ### https://data.binance.vision/data/spot/daily/klines/BTCFDUSD/1s/BTCFDUSD-1s-2023-09-23.zip
    if type == 'um' or type == 'cm':
        url = f'https://data.binance.vision/data/futures/{type}/monthly/{data}/{ticker}/{interval}/{ticker}-{interval}-'
        output_folder = DATA_DIR + f'binance_vision/futures_{type}/{data}/{ticker}{interval}'
    elif type == 'spot':
        url = f'https://data.binance.vision/data/{type}/monthly/{data}/{ticker}/{interval}/{ticker}-{interval}-'
        output_folder = DATA_DIR + f'binance_vision/{type}/{data}/{ticker}{interval}'
    else:
        raise ValueError("type argument must be one of { um, cm, spot }")
    print(f'output_folder: {output_folder}')

    if path.isfile(output_folder + '.csv'):
        df = pd.read_csv(output_folder + '.csv')
        df['Opened'] = pd.to_datetime(df['Opened'])
        last_timestamp = (df.iloc[-1]['Opened']).value // 10 ** 9
        if time() - last_timestamp > delay:
            start_date = pd.to_datetime(last_timestamp, unit='s').date()
            # print(f'start_date {start_date}')
            end_date = date.today()
            data_frames = [df]
        else:
            return df
    else:
        data_frames = collect_monthly(url, output_folder)
        end_date = date.today()
        start_date = date(end_date.year, end_date.month, 1)
    url = url.replace('monthly', 'daily')
    # print(url)
    ### binance vision does not store current day data so we substract 1 day
    while start_date <= end_date:
        _url = url + f'{start_date}.zip'
        output_path = path.join(output_folder, f'{start_date}')
        if path.exists(output_path) and listdir(output_path)[0].endswith('.csv'):
            data_frames.append(read_partial_df(output_path))
        else:
            print(f'downloading... {_url}')
            if download_and_unzip(_url, output_path):
                data_frames.append(read_partial_df(output_path))
            else:
                print('File does not exist.')
        start_date += timedelta(1)
    df = pd.concat(data_frames, ignore_index=True)
    # print(df)
    fixed_df = fix_and_fill_df(df, interval)
    fixed_df.to_csv(output_folder + '.csv', index=False)
    return fixed_df


def by_DataClient(ticker='BTCUSDT', interval='1m', futures=True, statements=True, delay=LAST_DATA_POINT_DELAY):
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
    return fix_and_fill_df(df, interval)
