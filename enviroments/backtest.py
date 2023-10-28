from collections import deque
# from random import normalvariate
from math import copysign, sqrt, floor
from random import randint
from time import time
from gym import spaces, Env
from numpy import array, inf, mean, std, random
from utility import TradingGraph


class SpotBacktest(Env):
    def __init__(self, df, dates_df=None, max_steps=0, exclude_cols_left=0,
                 init_balance=1_000, position_ratio=1.0, stop_loss=0.0, fee=0.0002, coin_step=0.001,
                 slippage=None, render_range=120, visualize=False):
        self.creation_t = time()
        print(f'Environment created. Fee: {fee} Coin step: {coin_step}')
        print(f' Obs sample (last row): {df[-1,:]}')
        print(f' Slippage stats: {slippage}')
        if visualize and dates_df is not None:
            self.dates_df = dates_df
            print(f' Visualization started, from: {self.dates_df[0]}', end=' ')
            print(f'to: {self.dates_df[-1]}')
            self.visualize = True
            self.render_range = render_range
        else:
            self.visualize = False
            print(f' Visualize is set to false or there was no dates df provided.')
        # This implementation uses only mean values provided by arg dict (slippage) #
        # as factor for calculation of real buy and sell prices. #
        # Generation of random numbers is too expensive computational wise. #
        # self.slippage = slippage
        if slippage is not None:
            self.buy_factor = slippage['market_buy'][0]
            self.sell_factor = slippage['market_sell'][0]
            self.stop_loss_factor = slippage['SL'][0]
        else:
            self.buy_factor, self.sell_factor, self.stop_loss_factor = 1.0, 1.0, 1.0
        self.df = df
        self.total_steps = len(self.df)
        self.exclude_cols_left = exclude_cols_left
        self.coin_step = coin_step
        self.fee = fee
        self.max_steps = max_steps
        self.init_balance = init_balance
        self.init_position_size = init_balance * position_ratio
        self.position_ratio = position_ratio
        self.position_size = self.init_position_size
        self.balance = self.init_balance
        self.stop_loss = stop_loss
        # Discrete action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)
        # Observation space #
        # none_df_obs_count are observations like current PnL, account balance, asset quantity etc. #
        other_obs_count = 0
        obs_space_dims = len(self.df[0, exclude_cols_left:])+other_obs_count
        obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        self.creation_t = time()
        self.total_steps = len(self.df)
        self.done = False
        self.reward = 0
        self.output = False
        if self.visualize:
            self.trades = deque(maxlen=self.render_range)
            self.visualization = TradingGraph(self.render_range)
        self.info = {}
        self.PL_ratios_and_PNLs = []
        self.balance = self.init_balance
        self.init_position_size = self.init_balance * self.position_ratio
        self.position_size = self.init_position_size
        self.prev_bal = 0
        self.enter_price = 0
        self.stop_loss_price = 0
        self.qty = 0
        self.pnl = 0
        self.SL_losses, self.cumulative_fees, self.liquidations = 0, 0, 0
        self.in_position, self.position_closed, self.in_position_counter, self.episode_orders = 0, 0, 0, 0
        self.good_trades_count, self.bad_trades_count = 1, 1
        self.profit_mean, self.loss_mean = 0, 0
        self.max_drawdown, self.max_profit = 0, 0
        self.loss_hold_counter, self.profit_hold_counter = 0, 0
        self.max_balance = self.min_balance = self.balance
        if self.max_steps > 0:
            self.start_step = randint(0, self.total_steps - self.max_steps)
            self.end_step = self.start_step + self.max_steps - 1
        else:
            self.start_step = 0
            self.end_step = self.total_steps - 1
        self.current_step = self.start_step
        self.obs = iter(self.df[self.start_step:self.end_step, self.exclude_cols_left:])
        return next(self.obs)

    def _next_observation(self):
        try:
            self.current_step += 1
            # _obs = next(self.obs)
            # print(_obs)
            return next(self.obs)
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            return self.df[self.current_step, self.exclude_cols_left:]

    '''def _next_observation(self):
      self.current_step += 1
      if self.current_step==self.end_step-1:
        self._finish_episode()
      return next(self.obs)'''

    '''def _random_factor(self, price, trade_type):
      return round(price*float(normalvariate(self.slippage[trade_type][0], self.slippage[trade_type][1])), 2)'''

    def _buy(self, price):
        self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
        # Considering random factor as in real world scenario #
        # price = self._random_factor(price, 'market_buy')
        price = price * self.buy_factor
        self.in_position = 1
        self.episode_orders += 1
        self.enter_price = price
        # When there is no fee, subtract 1 just to be sure balance can buy this amount #
        step_adj_qty = floor((self.position_size * (1 - 2 * self.fee)) / (price * self.coin_step))
        if step_adj_qty == 0:
            self._finish_episode()
        self.qty = round(step_adj_qty * self.coin_step, 5)
        self.position_size = round(self.qty * price, 2)
        self.prev_bal = self.balance
        self.balance -= self.position_size
        fee = (self.position_size * self.fee)
        self.position_size -= fee
        self.cumulative_fees += fee
        # print(f'BOUGHT {self.qty} at {price}')
        if self.visualize:
            self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': "open_long", 'Reward': 0})

    def _sell(self, price, sl=False):
        # print(f'SOLD {self.qty} at {price}')
        if sl:
            '''price = self._random_factor(price, 'SL')
        while price>self.enter_price:
          price = self._random_factor(price, 'SL')'''
            price = price * self.stop_loss_factor
            order_type = 'open_short'
        else:
            '''price = self._random_factor(price, 'market_sell')'''
            price = price * self.sell_factor
            order_type = 'close_long'
        self.balance += round(self.qty * price, 2)
        fee = abs(price * self.qty * self.fee)
        self.balance -= fee
        self.cumulative_fees += fee
        percentage_profit = (self.balance / self.prev_bal) - 1
        # PROFIT #
        if percentage_profit > 0:
            if self.balance >= self.max_balance: self.max_balance = self.balance
            self.good_trades_count += 1
            if self.max_profit == 0 or percentage_profit > self.max_profit:
                self.max_profit = percentage_profit
        # LOSS #
        elif percentage_profit < 0:
            if self.balance <= self.min_balance: self.min_balance = self.balance
            self.bad_trades_count += 1
            if self.max_drawdown == 0 or percentage_profit < self.max_drawdown:
                self.max_drawdown = percentage_profit
            if sl:
                self.SL_losses += self.prev_bal - self.balance
        self.PL_ratios_and_PNLs.append((percentage_profit, self.good_trades_count / self.bad_trades_count))
        self.position_size = (self.balance * self.position_ratio)
        # If balance minus position_size and fee is less or eq 0 #
        if self.position_size < (price * self.coin_step):
            self._finish_episode()
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.stop_loss_price = 0
        if self.visualize:
            self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': order_type, 'Reward': 0})

    def step(self, action):
        # 30-day DCA, adding 222USD to balance
        # if (self.current_step-self.start_step)%43_200 == 0:
        # self.balance+=222
        # self.init_balance+=222
        if self.in_position:
            low, close = self.df[self.current_step, 2:4]
            # print(f'low: {low}, close: {close}, self.enter_price: {self.enter_price}')
            self.in_position_counter += 1
            if close > self.enter_price:
                self.profit_hold_counter += 1
            else:
                self.loss_hold_counter += 1
            if low <= self.stop_loss_price:
                self._sell(self.stop_loss_price, sl=True)
            elif action == 2 and self.qty > 0:
                self._sell(close)
        elif action == 1:
            close = self.df[self.current_step, 3]
            self._buy(close)
        elif (not self.episode_orders) and ((self.current_step - self.start_step) > 2_880):
            self._finish_episode()
        '''if self.current_step==self.end_step:
        if self.in_position:
          close = self.df[self.current_step, 3]
          self._sell(close)
        return self._finish_episode()'''
        '''info = {'action': action,
                'in_position': self.in_position,
                'position_size': self.position_size,
                'balance': self.balance,
                'reward': 0,
                'current_step': self.current_step,
                'end_step': self.end_step}'''
        # print(info)
        return self._next_observation(), self.reward, self.done, self.info

    def render(self, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            _date = self.dates_df[self.current_step]
            _open = self.df[self.current_step, 0]
            _high = self.df[self.current_step, 1]
            _low = self.df[self.current_step, 2]
            _close = self.df[self.current_step, 3]
            # Volume = self.df[self.current_step, 4]
            _volume = self.df[self.current_step, -1]
            _dohlcv = [_date, _open, _high, _low, _close, _volume]
            if self.in_position:
                _pnl = self.enter_price / _close - 1
                self.visualization.render(_dohlcv,
                                          self.balance + (self.position_size + (self.position_size * _pnl)),
                                          self.trades)
            else:
                self.visualization.render(_dohlcv, self.balance, self.trades)

    def _finish_episode(self):
        # print('BacktestEnv._finish_episode()')
        if self.in_position:
            close = self.df[self.current_step, 3]
            self._sell(close)
        self.done = True
        # if (self.current_step==self.end_step) and self.good_trades_count>1 and self.bad_trades_count>1:
        if self.good_trades_count > 1 and self.bad_trades_count > 1:
            self.PNL_arrays = array(self.PL_ratios_and_PNLs)
            # self.realized_PNLs, self.PL_count_ratios = array([e[0] for e in self.PL_ratios_and_PNLs]), array([e[1] for e in self.PL_ratios_and_PNLs])
            gain = self.balance - self.init_balance
            total_return = (self.balance / self.init_balance) - 1
            risk_free_return = (self.df[-1, 3] / self.df[0, 3]) - 1
            PNL_mean, PNL_stdev = mean(self.PNL_arrays[:, 0]), std(self.PNL_arrays[:, 0])
            profit_mean = mean(self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] > 0])
            losses_mean = mean(self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0])
            losses_stdev = std(self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0])
            hold_ratio = self.profit_hold_counter / self.loss_hold_counter if self.loss_hold_counter > 0 and self.profit_hold_counter > 0 else 1
            self.PL_count_mean = mean(self.PNL_arrays[:, 1])
            self.PL_ratio = abs(profit_mean / losses_mean)
            # PL_count_final = self.good_trades_count/self.bad_trades_count
            # PLratio_x_PLcount = self.PL_ratio*self.PL_count_mean
            self.sharpe_ratio = (PNL_mean - risk_free_return) / PNL_stdev if PNL_stdev != 0 else -1
            self.sortino_ratio = (total_return - risk_free_return) / losses_stdev if losses_stdev != 0 else -1
            # self.reward = copysign((abs(gain)**1.5)*self.PL_count_mean*sqrt(hold_ratio)*sqrt(self.PL_ratio)*sqrt(self.episode_orders), gain)/self.total_steps
            # self.reward = copysign(gain**2, gain)+(self.episode_orders/sqrt(self.total_steps))+self.PL_count_mean+sqrt(hold_ratio)+sqrt(self.PL_ratio)
            self.reward = (copysign(abs(gain) ** 1.5, gain) * self.episode_orders * (self.PL_count_mean ** 2) * sqrt(
                hold_ratio) * sqrt(self.PL_ratio)) / self.total_steps
            # print(self.reward)
            slope_indicator = 1.000
            '''slope_indicator = linear_slope_indicator(self.PL_count_ratios)
        if self.reward<0 and slope_indicator<0:
          self.reward = self.reward*slope_indicator*-1
        else:
          self.reward = self.reward*slope_indicator'''
            # if gain>0:
            if gain > self.init_balance * .1:
            # if True:
                self.output = True
                print(
                    f'Episode finished: gain:${gain:.2f}, cumulative_fees:${self.cumulative_fees:.2f}, SL_losses:${self.SL_losses:.2f}, liquidations:{self.liquidations}')
                print(
                    f' episode_orders:{self.episode_orders:_}, trades_count(profit/loss):{self.good_trades_count:_}/{self.bad_trades_count:_}, trades_avg(profit/loss):{profit_mean * 100:.2f}%/{losses_mean * 100:.2f}%, ',
                    end='')
                print(f'max(profit/drawdown):{self.max_profit * 100:.2f}%/{self.max_drawdown * 100:.2f}%')
                print(
                    f' reward:{self.reward:.8f}, PL_ratio:{self.PL_ratio:.3f}, PL_count_mean:{self.PL_count_mean:.3f}, hold_ratio:{hold_ratio:.3f}, PNL_mean:{PNL_mean * 100:.2f}%')
                print(
                    f' slope_indicator:{slope_indicator:.4f}, sharpe_ratio:{self.sharpe_ratio:.2f}, sortino_ratio:{self.sortino_ratio:.2f}')
            else:
                self.output = False
            self.info = {'gain': gain,
                         'PL_ratio': self.PL_ratio,
                         'PL_count_mean': self.PL_count_mean,
                         'hold_ratio': hold_ratio,
                         'PNL_mean': PNL_mean,
                         'slope_indicator': slope_indicator,
                         'exec_time': time() - self.creation_t}
        else:
            self.output = False
            self.reward = -inf
            self.sharpe_ratio, self.sortino_ratio, self.PL_count_mean, self.PL_ratio = -1, -1, -1, -1
            self.info = {'gain': 0, 'PL_ratio': 0, 'hold_ratio': 0,
                         'PL_count_mean': 0, 'PNL_mean': 0, 'slope_indicator': 0,
                         'exec_time': time() - self.creation_t}
            # print(f'EPISODE FAILED! (end_step not reached OR profit/loss trades less than 2)')
        # return self.df[self.current_step, self.exclude_count:], self.reward, self.done, self.info


######################################################################################################################
######################################################################################################################
######################################################################################################################
class FuturesBacktest(SpotBacktest):
    def __init__(self, df, dates_df=None, df_mark=None, excluded_left=0, init_balance=1_000, position_ratio=1.0,
                 leverage=1, StopLoss=0.0, fee=0.0002, coin_step=0.001,
                 slippage={'market_buy': (1.0, 0.0), 'market_sell': (1.0, 0.0), 'SL': (1.0, 0.0)}, max_steps=0,
                 lookback_window_size=0, render_range=120, visualize=False):
        super().__init__(df=df, dates_df=dates_df, excluded_left=excluded_left, init_balance=init_balance,
                         position_ratio=position_ratio, StopLoss=StopLoss, fee=fee, coin_step=coin_step,
                         slippage=slippage, max_steps=max_steps, lookback_window_size=lookback_window_size,
                         render_range=render_range, visualize=visualize)
        # https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
        self.POSITION_TIER = {1: (125, .0040, 0), 2: (105, .005, 50),
                              3: (50, .01, 1_300), 4: (20, .025, 46_300),
                              5: (10, .05, 546_300), 6: (5, .10, 2_546_300),
                              7: (4, .125, 5_046_300), 8: (3, .15, 8_046_300),
                              9: (2, .25, 28_046_300), 10: (1, .50, 103_046_300)}
        self.leverage = leverage
        self.df_mark = df_mark
        # BTCUSDTperp last 1Y mean=6.09e-05 stdev=6.52e-05, mean+2*stedv covers ~95,4% of variance
        # self.funding_rate = 0.01912 * (1/100)
        self.funding_rate = 0.01 * (1 / 100)

    def reset(self, **kwargs):
        self.margin = 0
        self.liquidation_price = 0
        self.tier = 0
        # self.stop_loss /= self.leverage
        self.liquidations = 0
        return super().reset()
        '''for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            #self.orders_history.append([self.position_size, self.balance, self.in_position, copysign(1, self.qty), self.df[current_step, 4]-self.enter_price])
            self.market_history.append(self.df[self.current_step, self.exclude_count:])
        # backtest #
        return self.market_history.pop()
        #return np.concatenate((self.market_history, self.orders_history), axis=1)'''

    def _check_tier(self):
        # print('_check_tier')
        if self.position_size < 50_000:
            self.tier = 1
        elif 50_000 < self.position_size < 250_000:
            self.tier = 2
        elif 250_000 < self.position_size < 3_000_000:
            self.tier = 3
        elif 3_000_000 < self.position_size < 15_000_000:
            self.tier = 4
        elif 15_000_000 < self.position_size < 30_000_000:
            self.tier = 5
        elif 30_000_000 < self.position_size < 80_000_000:
            self.tier = 6
        elif 80_000_000 < self.position_size < 100_000_000:
            self.tier = 7
        elif 100_000_000 < self.position_size < 200_000_000:
            self.tier = 8
        elif 200_000_000 < self.position_size < 300_000_000:
            self.tier = 9
        elif 300_000_000 < self.position_size < 500_000_000:
            self.tier = 10

    '''def _check_margin(self):
      #print('_check_margin')
      if self.qty>0:
        min_price = self.df[self.current_step, 2]
      elif self.qty<0:
        min_price = self.df[self.current_step, 1]
      else:
        pass
        #print('co Ty tu robisz?')
      position_value = abs(self.qty*min_price)
      unrealized_PNL = abs(self.qty*self.enter_price/self.leverage)*self._get_pnl(min_price)
      # 1.25% Liquidation Clearance
      margin_balance = self.margin + unrealized_PNL - (position_value*0.0125) - (position_value*self.fee)
      maintenance_margin = position_value*self.POSITION_TIER[self.tier][1]-self.POSITION_TIER[self.tier][2]
      print(f'min_price:{min_price:.2f} position_value:{position_value:.2f} unrealized_PNL:{unrealized_PNL:.2f} Clearance:{(position_value*0.0125)} fee:{(position_value*self.fee)} margin:{self.margin} margin_balance:{margin_balance:.2f} maintenance_margin:{maintenance_margin:.2f} margin_ratio:{maintenance_margin/margin_balance*100}')
      if maintenance_margin>margin_balance:
        return True
      else:
        return False'''

    def _check_margin(self):
        # print('_check_margin')
        self.liquidation_price = (self.margin - self.qty * self.enter_price) / (
                    abs(self.qty) * self.POSITION_TIER[self.tier][1] - self.qty)
        # print(f'liquidation_price:{self.liquidation_price} (margin:{self.margin} qty:{self.qty} enter_price:{self.enter_price})')
        if self.qty > 0:
            min_price = self.df_mark[self.current_step, 2]
            if self.liquidation_price >= min_price: return True
        elif self.qty < 0:
            max_price = self.df_mark[self.current_step, 1]
            if self.liquidation_price <= max_price: return True
        else:
            return False

    def _get_pnl(self, price, update=False):
        _pnl = (((price / self.enter_price) - 1) * self.leverage) * copysign(1, self.qty)
        if update:
            self.pnl = _pnl
        elif not self.in_position:
            self.pnl = 0
        return _pnl

    def _open_position(self, side, price):
        if side == 'long':
            if self.visualize:
                self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                    'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': "open_long", 'Reward': 0})
                print(f'OPENING LONG at {price}')
            rnd_factor = random.normal(self.slippage['market_buy'][0], self.slippage['market_buy'][1], 1)[0]
            price = round(price * rnd_factor, 2)
            self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
        elif side == 'short':
            if self.visualize:
                self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                    'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': "open_short", 'Reward': 0})
                print(f'OPENING SHORT at {price}')
            rnd_factor = random.normal(self.slippage['market_sell'][0], self.slippage['market_sell'][1], 1)[0]
            price = round(price * rnd_factor, 2)
            self.stop_loss_price = round((1 + self.stop_loss) * price, 2)
        # print(f'OPENING x{self.leverage} {side} at price {price} ')
        self.in_position = 1
        self.episode_orders += 1
        self.enter_price = price
        self._check_tier()
        if self.leverage > self.POSITION_TIER[self.tier][0]:
            print(
                f' Leverage exceeds tier {self.tier} max, changing from {self.leverage} to {self.POSITION_TIER[self.tier][0]} (Balance: ${self.balance})')
            # print(f'Balance: {self.balance}')
            self.leverage = self.POSITION_TIER[self.tier][0]
        adj_qty = floor(self.position_size * self.leverage / (price * self.coin_step))
        if adj_qty == 0:
            adj_qty = 1
            # print('Forcing adj_qty to 1. Calculated quantity possible to buy with given postion_size and coin_step equals 0')
        self.margin = (adj_qty * self.coin_step * price) / self.leverage
        # print(f'  (position size {self.position_size} qty {adj_qty})')
        if self.margin > self.balance: return self._finish_episode()
        self.prev_bal = self.balance
        self.balance -= self.margin
        fee = (self.margin * self.fee * self.leverage)
        # print(f'  (fee {fee:.8f})')
        self.margin -= fee
        # print(f'OPENING POSITION fee:{fee:.2f} Margin:{self.margin:.2f} Balance:{self.balance+self.margin:.2f}')
        self.cumulative_fees += fee
        # print(f' balance(minus position size and fee) {self.balance}')
        if side == 'long':
            self.qty = adj_qty * self.coin_step
        elif side == 'short':
            self.qty = -1 * adj_qty * self.coin_step
        # sleep(2)

    def _close_position(self, price, liquidated=False, SL=False):
        if SL:
            rnd_factor = random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
            if self.qty > 0:
                while price * rnd_factor > self.enter_price:
                    rnd_factor = random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
            elif self.qty < 0:
                while price * rnd_factor < self.enter_price:
                    rnd_factor = random.normal(self.slippage['SL'][0], self.slippage['SL'][1], 1)[0]
        else:
            if self.qty > 0:
                rnd_factor = random.normal(self.slippage['market_sell'][0], self.slippage['market_sell'][1], 1)[0]
            elif self.qty < 0:
                rnd_factor = random.normal(self.slippage['market_buy'][0], self.slippage['market_buy'][1], 1)[0]
        if self.visualize:
            if self.qty > 0:
                if SL:
                    trade_type = "SL_long"
                elif liquidated:
                    trade_type = "liquidate_long"
                else:
                    trade_type = "close_long"
                print(f'CLOSING LONG at {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
            elif self.qty < 0:
                if SL:
                    trade_type = "SL_short"
                elif liquidated:
                    trade_type = "liquidate_short"
                else:
                    trade_type = "close_short"
                print(f'CLOSING SHORT at {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
            self.trades.append({'Date': self.dates_df[self.current_step], 'High': self.df[self.current_step, 1],
                                'Low': self.df[self.current_step, 2], 'total': self.qty, 'type': trade_type})
        price = round(price * rnd_factor, 2)
        # print(f'CLOSING position at price {price} liquidated:{liquidated} SL:{SL} SL_price:{self.stop_loss_price}')
        _position_value = abs(self.qty) * price
        _fee = (_position_value * self.fee)
        if liquidated:
            margin_balance = 0
        else:
            unrealized_PNL = (abs(self.qty) * self.enter_price / self.leverage) * self.pnl
            margin_balance = self.margin + unrealized_PNL - _fee
        # print(f'CLOSING POSITION fee:{_fee:.2f} UPNL:{unrealized_PNL:.2f} MarginBalance:{margin_balance:.2f} Balance:{self.balance:.2f}')
        self.cumulative_fees += _fee
        self.balance += margin_balance
        self.margin = 0
        percentage_profit = (self.balance / self.prev_bal) - 1
        ### PROFIT
        if percentage_profit > 0:
            if self.balance >= self.max_balance:
                self.max_balance = self.balance
                if self.balance > 1_000_000:
                    print(f'$$$$$$$$ {self.balance} $$$$$$$$')
            self.good_trades_count += 1
            if self.max_profit == 0 or percentage_profit > self.max_profit:
                self.max_profit = percentage_profit
        ### LOSS
        elif percentage_profit < 0:
            if self.balance <= self.min_balance: self.min_balance = self.balance
            self.bad_trades_count += 1
            if self.max_drawdown == 0 or percentage_profit < self.max_drawdown:
                self.max_drawdown = percentage_profit
            if SL:
                self.SL_losses += (self.balance - self.prev_bal)
        self.PL_ratios_and_PNLs.append((percentage_profit, self.good_trades_count / self.bad_trades_count))
        self.position_size = (self.balance * self.position_ratio)
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.stop_loss_price = 0
        self.pnl = 0
        # sleep(2)

    # Execute one time step within the environment
    def step(self, action):
        # print(f'current_step:{self.current_step} start_step:{self.start_step} end_step:{self.end_step} margin:{self.margin} balance:{self.balance}')
        close = self.df[self.current_step, 3]
        if self.current_step == self.end_step:
            if self.in_position: self._close_position(close)
            return self._finish_episode()
        if not self.in_position:
            if action == 0:
                if ((self.current_step - self.start_step) > 10_080) and (not self.episode_orders):
                    # print(f'(episode finished: {self.episode_orders} trades {self.current_step-self.start_step} steps)')
                    return self._finish_episode()
            elif self.position_size <= ((close * self.coin_step) / self.leverage) + (
                    (close * self.coin_step) * self.fee):
                return self._finish_episode()
            elif action == 1:
                self._open_position('long', close)
            elif action == 2:
                self._open_position('short', close)
        elif self.in_position:
            self.in_position_counter += 1
            if self.in_position_counter % 480 == 0:
                self.margin -= (abs(self.qty) * close * self.funding_rate)
            self._get_pnl(close, update=True)
            if self.pnl < 0:
                self.loss_hold_counter += 1
            elif self.pnl > 0:
                self.profit_hold_counter += 1
            if (self.df[self.current_step, 2] <= self.stop_loss_price and self.qty > 0) or (
                    self.df[self.current_step, 1] >= self.stop_loss_price and self.qty < 0):
                self._close_position(self.stop_loss_price, SL=True)
            elif self._check_margin():
                self.liquidations += 1
                self._close_position(close, liquidated=True)
            else:
                if action == 0:
                    pass
                elif (action == 1 and self.qty < 0) or (action == 2 and self.qty > 0):
                    self._close_position(close)
        info = {'action': action,
                'reward': 0,
                'step': self.current_step,
                'exec_time': time() - self.creation_t}
        self.current_step += 1
        # sleep(0.1)
        return self._next_observation(), 0, self.done, info
