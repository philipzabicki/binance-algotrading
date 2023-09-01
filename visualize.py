#import pandas as pd
#import numpy as np
import datetime as dt
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
#from datetime import datetime
import os
#import cv2

def Write_to_file(list_values, filename='{}.txt'.format(dt.datetime.now().strftime("%Y-%m-%d"))):
  line=''
  for i in list_values:
    #print(i)
    line += "{},".format(str(i))
    #print(Date)
  if not os.path.exists('logs'):
    os.makedirs('logs')
  file = open("logs\\"+filename, 'a+')
  file.write(line+"\n")
  file.close()
  
class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, Render_range):
        #self.Volume = deque(maxlen=Render_range)
        #self.net_worth = deque(maxlen=Render_range)
        self.render_data = deque(maxlen=Render_range)
        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8)) 
        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()
        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%Y-%m-%d %H:%M')
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.set_ylabel('Balance')
        #self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        # Add paddings to make graph easier to view
        #plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)
    # Render the environment to the screen
    def render(self, Date, Open, High, Low, Close, Volume, net_worth, trades):
        Date = mpl_dates.date2num([Date])[0]
        self.render_data.append([Date, Open, High, Low, Close, Volume, net_worth])
        render_data_zip=list(zip(*self.render_data))
        tohlc = [list(elem) for elem in zip(*render_data_zip[0:5])]
        #print(render_data_zip)
        #time.sleep(5)
        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, tohlc, width=0.015/24, colorup='green', colordown='red', alpha=0.8)
        # Put all dates to one list and fill ax2 sublot with volume
        Date_Render_range = render_data_zip[0]
        self.ax2.clear()
        self.ax2.fill_between(Date_Render_range, render_data_zip[5], 0)
        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(Date_Render_range, render_data_zip[6], color="blue")
        # beautify the x-labels (Our Date format)
        self.fig.autofmt_xdate()
        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([trade['Date']])[0]
            if trade_date in Date_Render_range:
                if trade['type'] == 'open_long':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='white', label='white', s = 180, edgecolors='green', marker="^")
                elif trade['type'] == 'open_short':
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='white', label='white', s = 180, edgecolors='red', marker="v")
                elif trade['type'] == 'close_long':
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='black', label='black', s = 180, edgecolors='red', marker="v")
                elif trade['type'] == 'liquidate_long':
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 180, edgecolors='red', marker="v")
                elif trade['type'] == 'SL_long':
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='orange', label='orange', s = 180, edgecolors='orange', marker="v")
                elif trade['type'] == 'close_short':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='black', label='black', s = 180, edgecolors='green', marker="^")
                elif trade['type'] == 'liquidate_short':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 180, edgecolors='red', marker="^")
                elif trade['type'] == 'SL_short':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='orange', label='orange', s = 180, edgecolors='orange', marker="^")
        # we need to set layers every step, because we are clearing subplots every step
        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()
        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        plt.show(block=False)
        # Necessary to view frames before they are unrendered
        plt.pause(0.0000001)
        """Display image with OpenCV - no interruption"""
        '''# redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot",image)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return'''