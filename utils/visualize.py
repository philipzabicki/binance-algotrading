from collections import deque

from cv2 import cvtColor, imshow, waitKey, destroyAllWindows, COLOR_RGB2BGR
from matplotlib import dates as mpl_dates, pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
from numpy import array, where, max, min, fromstring, uint8


class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, render_range, time_step=1 / 24, Show_reward=True, Show_indicators=False):
        self.render_queue = deque(maxlen=render_range)
        self.render_arr = array(self.render_queue)
        self.trades = deque(maxlen=render_range)
        self.trades_arr = array(self.trades)
        self.render_range = render_range
        self.time_step = float(time_step) * .8
        # print(f'self.time_step {self.time_step}')
        self.date_format = mpl_dates.DateFormatter('%Y-%m-%d\n%H:%M:%S')

        self.Show_reward = Show_reward
        self.Show_indicators = Show_indicators

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16, 9), dpi=75)

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=5, colspan=1)

        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=self.ax1)

        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        # self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')

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

    # Render the environment to the screen
    # def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):

    def append(self, data_portion, trade):
        # print(f'type(data_portion[0]) {type(data_portion[0])}')
        # data_portion[0] = mpl_dates.num2date(data_portion[0])
        # print(f'_date {type(_date)}')
        # print(f'data_portion {data_portion}')
        self.render_queue.append(data_portion)
        self.trades.append(trade)
        self.render_arr = array(self.render_queue)
        self.trades_arr = array(self.trades)
        # print(f'self.trades_arr {self.trades_arr}')

    def render(self):
        # print(f'render_arr[:, 0:5] {self.render_arr[:, 0:5]}')
        ohlc_equal = where(self.render_arr[:, 2] == self.render_arr[:, 3])
        # print(f'ohlc_equal {ohlc_equal}')
        # print(self.render_arr[ohlc_equal, 2])
        self.render_arr[ohlc_equal, 2] *= 1.00001
        self.render_arr[ohlc_equal, 3] *= 0.99999

        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_arr, width=self.time_step, colorup='green', colordown='red', alpha=1.0)

        # Put all dates to one list and fill ax2 sublot with volume
        # print(f'Date_Render_range {Date_Render_range}')
        # ploting indicator/reward
        self.ax2.clear()
        self.ax2.fill_between(self.render_arr[:, 0], self.render_arr[:, 5], 0)

        # if self.Show_indicators:
        # self.plot_indicators(df, Date_Render_range)

        # draw our balance graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(self.render_arr[:, 0], self.render_arr[:, 6], color="blue")

        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        RANGE = max(self.render_arr[:, 2]) - min(self.render_arr[:, 3])

        # sort sell and buy orders, put arrows in appropiate order positions
        idx = list(*where('' != self.trades_arr[:, 0]))
        # print(f'idx {idx}')
        for i in idx:
            # print(f'i: {i} self.trades_arr[i]: {self.trades_arr[i]}')
            annotate_text = '$' + self.trades_arr[i, 1] if (self.trades_arr[i, 1] != '0.0') and (
                    self.trades_arr[i, 1] != '-0.0') else ''
            x_position = self.render_arr[i, 0]
            if self.trades_arr[i, 0] == 'open_long' or self.trades_arr[i, 0] == 'close_short' or self.trades_arr[
                i, 0] == 'take_profit_short':
                high_low = self.render_arr[i, 3] - RANGE * 0.02
                ycoords = self.render_arr[i, 3] - RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='green', label='green', s=self.render_range * .55,
                                 edgecolors='black',
                                 marker="^")
                self.ax1.annotate(annotate_text, (x_position, high_low * 0.993), c='black', ha='center', va='center',
                                  size=self.render_range * .04)
            elif self.trades_arr[i, 0] == 'open_short' or self.trades_arr[i, 0] == 'close_long' or self.trades_arr[
                i, 0] == 'take_profit_long':
                high_low = self.render_arr[i, 2] + RANGE * 0.02
                ycoords = self.render_arr[i, 2] + RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='red', label='red', s=self.render_range * .5,
                                 edgecolors='black',
                                 marker="v")
                self.ax1.annotate(annotate_text, (x_position, high_low * 1.008), c='black', ha='center', va='center',
                                  size=self.render_range * .04)
            elif self.trades_arr[i, 0] == 'stop_loss_long':
                high_low = self.render_arr[i, 3] - RANGE * 0.02
                ycoords = self.render_arr[i, 3] - RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='yellow', label='yellow', s=self.render_range * .6,
                                 edgecolors='black',
                                 marker="d")
                self.ax1.annotate(annotate_text, (x_position, high_low * 0.993), c='black', ha='center', va='center',
                                  size=self.render_range * .04)
            elif self.trades_arr[i, 0] == 'stop_loss_short':
                high_low = self.render_arr[i, 2] + RANGE * 0.02
                ycoords = self.render_arr[i, 2] + RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='yellow', label='yellow', s=self.render_range * .6,
                                 edgecolors='black',
                                 marker="d")
                self.ax1.annotate(annotate_text, (x_position, high_low * 1.007), c='black', ha='center', va='center',
                                  size=self.render_range * .04)
            elif self.trades_arr[i, 0] == 'liquidate_long':
                high_low = self.render_arr[i, 3] - RANGE * 0.02
                ycoords = self.render_arr[i, 3] - RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='black', label='black', s=self.render_range * .6,
                                 edgecolors='black',
                                 marker="X")
                self.ax1.annotate(annotate_text, (x_position, high_low * 0.993), c='black', ha='center', va='center',
                                  size=self.render_range * .04)
            elif self.trades_arr[i, 0] == 'liquidate_short':
                high_low = self.render_arr[i, 2] + RANGE * 0.02
                ycoords = self.render_arr[i, 2] + RANGE * 1.10
                self.ax1.scatter(x_position, high_low, c='black', label='black', s=self.render_range * .6,
                                 edgecolors='black',
                                 marker="X")
                self.ax1.annotate(annotate_text, (x_position - 2 * self.time_step, high_low * 1.007), c='black',
                                  ha='center', va='center', size=self.render_range * .04)
            '''try:
                self.ax1.annotate('{0:.2f}'.format(self.render_arr[i, 5]), (self.render_arr[i, 0] - 0.02, high_low),
                                  xytext=(self.render_arr[i, 0] - 0.02, ycoords),
                                  bbox=dict(boxstyle='round', fc='w', ec='k', lw=1), fontsize="small")
            except Exception as e:
                print(f'Exception: {e}')'''

        # we need to set layers every step, because we are clearing subplots every step
        # box1 = dict(facecolor="blue", pad=2, alpha=0.4)
        # box2 = dict(facecolor="red", pad=2, alpha=0.4)
        self.ax2.set_xlabel('Date')
        self.ax2.set_ylabel('Indicator/Reward', fontsize=16)
        self.ax2.yaxis.label.set_color('red')
        self.ax1.set_ylabel('Price', fontsize=16)
        self.ax3.yaxis.set_label_position("right")
        self.ax3.set_ylabel('Balance', fontsize=16)
        self.ax3.yaxis.label.set_color('blue')

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
        img = fromstring(self.fig.canvas.tostring_rgb(), dtype=uint8, sep='')
        img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        image = cvtColor(img, COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        imshow("Bitcoin trading bot", image)

        if waitKey(25) & 0xFF == ord("q"):
            destroyAllWindows()
            return
        else:
            return img
