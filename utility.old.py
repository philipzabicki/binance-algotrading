import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
matplotlib.use('Agg')
from dateutil.parser import parse
from scipy.stats import skew, kurtosis
from pympler import asizeof
import matplotlib.dates as mpl_dates
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from collections import deque
from definitions import ROOT_DIR
import cv2


def get_market_slips_stats():
    buy = pd.read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    sell = pd.read_csv(ROOT_DIR + '/settings/slippages_market_sell.csv')
    SL = pd.read_csv(ROOT_DIR + '/settings/slippages_StopLoss.csv')
    slipps = {'market_buy': (buy.values.mean(), buy.values.std()),
              'market_sell': (sell.values.mean(), sell.values.std()), 'SL': (SL.values.mean(), SL.values.std())}
    # print(slipps)
    return slipps


def get_limit_slips_stats():
    buy = pd.read_csv(ROOT_DIR + '/settings/slippages_limit_buy.csv')
    sell = pd.read_csv(ROOT_DIR + '/settings/slippages_limit_sell.csv')
    SL = pd.read_csv(ROOT_DIR + '/settings/slippages_StopLoss.csv')
    slipps = {'market_buy': (buy.values.mean(), buy.values.std()),
              'market_sell': (sell.values.mean(), sell.values.std()), 'SL': (SL.values.mean(), SL.values.std())}
    # print(slipps)
    return slipps


'''def get_stats_for_file(file_path):
    df = pd.read_csv(file_path, header=0)
    mean = float(df.mean().values[0])
    std = float(df.std().values[0])
    return mean, std

def get_slips_stats():
    base_path = ROOT_DIR + '\\settings\\'
    file_names = ['slippages_market_buy.csv', 'slippages_market_sell.csv', 'slippages_StopLoss.csv']
    labels = ['market_buy', 'market_sell', 'SL']
    stats = { label:get_stats_for_file(base_path + file_name) for label,file_name in zip(labels,file_names) }
    return stats'''

'''def get_slips_stats():
    buy = pd.read_csv(ROOT_DIR+'/settings/slippages_market_buy.csv', header=0)
    sell = pd.read_csv(ROOT_DIR+'/settings/slippages_market_sell.csv', header=0)
    SL = pd.read_csv(ROOT_DIR+'/settings/slippages_StopLoss.csv', header=0)
    return {'market_buy':(float(np.mean(buy)), float(np.std(buy))),
            'market_sell':(float(np.mean(sell)), float(np.std(sell))),
            'SL':(float(np.mean(SL)), float(np.std(SL)))}'''


def get_slips_stats_advanced():
    buy = pd.read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    sell = pd.read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    SL = pd.read_csv(ROOT_DIR + '/settings/slippages_market_buy.csv')
    return {
        'market_buy': {
            'mean': buy.mean(),
            'std': buy.std(),
            'skewness': buy.apply(skew),
            'kurtosis': buy.apply(kurtosis)
        },
        'market_sell': {
            'mean': sell.mean(),
            'std': sell.std(),
            'skewness': sell.apply(skew),
            'kurtosis': sell.apply(kurtosis)
        },
        'SL': {
            'mean': SL.mean(),
            'std': SL.std(),
            'skewness': SL.apply(skew),
            'kurtosis': SL.apply(kurtosis)
        }
    }


# Calculates and returns linear regression slope but predictor variable(X) are natural numbers from 1 to len of dependent variable(Y)
# Y are supposed to be balance divided by initial balance ratios per every env step
def linear_reg_slope(Y):
    Y = np.array(Y)
    n = len(Y)
    X = np.arange(1, n + 1)
    # print(f'X: {X}')
    x_mean = np.mean(X)
    Sxy = np.sum(X * Y) - n * x_mean * np.mean(Y)
    Sxx = np.sum(X * X) - n * x_mean ** 2
    return Sxy / Sxx


def minutes_since(data_string):
    diff = dt.datetime.now() - parse(data_string, dayfirst=True)
    minutes = diff.total_seconds() / 60
    return int(minutes)


def seconds_since(data_string):
    diff = dt.datetime.now() - parse(data_string, dayfirst=True)
    seconds = diff.total_seconds()
    return int(seconds)


def get_attributes_and_deep_sizes(obj):
    attributes_and_sizes = {}
    for attribute_name in dir(obj):
        attribute_value = getattr(obj, attribute_name)
        _size = asizeof.asizeof(attribute_value)
        if _size > 1_000:
            attributes_and_sizes[attribute_name] = asizeof.asizeof(attribute_value)
    return attributes_and_sizes


class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, Render_range, Show_reward=True, Show_indicators=False):
        self.Volume = deque(maxlen=Render_range)
        self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        self.Render_range = Render_range
        self.Show_reward = Show_reward
        self.Show_indicators = Show_indicators

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16, 8))

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        # self.date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M:%S')

        # Add paddings to make graph easier to view
        # plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

        # define if show indicators
        if self.Show_indicators:
            self.Create_indicators_lists()

    def Create_indicators_lists(self):
        # Create a new axis for indicatorswhich shares its x-axis with volume
        self.ax4 = self.ax2.twinx()

        self.sma7 = deque(maxlen=self.Render_range)
        self.sma25 = deque(maxlen=self.Render_range)
        self.sma99 = deque(maxlen=self.Render_range)

        self.bb_bbm = deque(maxlen=self.Render_range)
        self.bb_bbh = deque(maxlen=self.Render_range)
        self.bb_bbl = deque(maxlen=self.Render_range)

        self.psar = deque(maxlen=self.Render_range)

        self.MACD = deque(maxlen=self.Render_range)
        self.RSI = deque(maxlen=self.Render_range)

    def plot_indicators(self, df, Date_Render_range):
        self.sma7.append(df["sma7"])
        self.sma25.append(df["sma25"])
        self.sma99.append(df["sma99"])

        self.bb_bbm.append(df["bb_bbm"])
        self.bb_bbh.append(df["bb_bbh"])
        self.bb_bbl.append(df["bb_bbl"])

        self.psar.append(df["psar"])

        self.MACD.append(df["MACD"])
        self.RSI.append(df["RSI"])

        # Add Simple Moving Average
        self.ax1.plot(Date_Render_range, self.sma7, '-')
        self.ax1.plot(Date_Render_range, self.sma25, '-')
        self.ax1.plot(Date_Render_range, self.sma99, '-')

        # Add Bollinger Bands
        self.ax1.plot(Date_Render_range, self.bb_bbm, '-')
        self.ax1.plot(Date_Render_range, self.bb_bbh, '-')
        self.ax1.plot(Date_Render_range, self.bb_bbl, '-')

        # Add Parabolic Stop and Reverse
        self.ax1.plot(Date_Render_range, self.psar, '.')

        self.ax4.clear()
        # # Add Moving Average Convergence Divergence
        self.ax4.plot(Date_Render_range, self.MACD, 'r-')

        # # Add Relative Strength Index
        self.ax4.plot(Date_Render_range, self.RSI, 'g-')
    def append(self):
         pass
    # Render the environment to the screen
    # def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
    def render(self, dohlcv, net_worth, trades):
        Date = dohlcv[0]
        Open = dohlcv[1]
        High = dohlcv[2]
        Low = dohlcv[3]
        Close = dohlcv[4]
        Volume = dohlcv[5]
        if Open == Close:
            High *= 1.00001
            Low *= 0.99999
        # append volume and net_worth to deque list
        self.Volume.append(Volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        # print(f'Date: {Date}')
        _Date = mpl_dates.date2num(Date)
        # print(f'_Date: {_Date}')
        self.render_data.append([_Date, Open, High, Low, Close])
        #print(f'self.render_data {self.render_data}')
        if len(self.render_data) > 1:
            difference = (self.render_data[1][0] - self.render_data[0][0])*.8
            print(f'timedelta: {difference}')
        else:
            difference = 1 / 100_000

        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=difference, colorup='green', colordown='red', alpha=1.0)

        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = [i[0] for i in self.render_data]
        # print(f'Date_Render_range {Date_Render_range}')
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, self.Volume, 0)

        # if self.Show_indicators:
            # self.plot_indicators(df, Date_Render_range)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, self.net_worth, color="blue")

        # beautify the x-labels (Our Date format)
        # self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        minimum = np.min(np.array(self.render_data)[:, 1:])
        # print(f'minimum {minimum}')
        maximum = np.max(np.array(self.render_data)[:, 1:])
        # print(f'maximum {maximum}')
        RANGE = maximum - minimum

        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num(trade['Date'])
            # print(f'trade_date {trade_date}')
            if trade_date in Date_Render_range:
                if trade['type'] == 'open_long' or trade['type'] == 'close_short':
                    high_low = trade['Low'] - RANGE * 0.02
                    ycoords = trade['Low'] - RANGE * 0.08
                    self.ax1.scatter(trade_date, high_low, c='green', label='green', s=120, edgecolors='none',
                                     marker="^")
                else:
                    high_low = trade['High'] + RANGE * 0.02
                    ycoords = trade['High'] + RANGE * 0.06
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s=120, edgecolors='none', marker="v")

                if self.Show_reward:
                    try:
                        # print(trade['Reward'])
                        self.ax1.annotate('{0:.2f}'.format(trade['Reward']), (trade_date - 0.02, high_low),
                                          xytext=(trade_date - 0.02, ycoords),
                                          bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
                    except Exception as e:
                        print(e)

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')

        # I use tight_layout to replace plt.subplots_adjustx
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        # plt.show(block=False)
        # Necessary to view frames before they are unrendered
        # plt.pause(0.001)

        """Display image with OpenCV - no interruption"""

        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.to_string_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot", image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        else:
            return img
