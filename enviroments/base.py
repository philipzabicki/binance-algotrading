# from random import normalvariate
from csv import writer
from datetime import datetime as dt
from math import copysign, floor, sqrt
from random import randint
from time import time

from gym import spaces, Env
from matplotlib.dates import date2num
from numpy import array, mean, std, inf, searchsorted
from pandas import to_datetime

from definitions import REPORT_DIR
from utils.visualize import TradingGraph


class SpotBacktest(Env):
    def __init__(self, df, start_date='', end_date='', max_steps=0, exclude_cols_left=1, no_action_finish=2_880,
                 init_balance=1_000, position_ratio=1.0, save_ratio=None, stop_loss=None, take_profit=None,
                 fee=0.0002, coin_step=0.001, slippage=None, slipp_std=0,
                 visualize=False, render_range=120, verbose=True, write_to_file=False, *args, **kwargs):
        self.creation_t = time()
        self.df = df.to_numpy()
        if start_date != '' and end_date != '':
            start_date, end_date = to_datetime(start_date), to_datetime(end_date)
            self.start_index = searchsorted(self.df[:, 0], start_date, side='left')
            self.end_index = searchsorted(self.df[:, 0], end_date, side='right') - 1
        else:
            self.start_index = 0
            self.end_index = self.df.shape[0] - 1
        trade_range_size = len(self.df[self.start_index:self.end_index, :])
        if len(self.df[self.start_index:self.end_index, :]) < max_steps:
            raise ValueError("max_steps larger than rows in dataframe")
        print(f'Environment ({self.__class__.__name__}) created.')
        print(f' fee:{fee}, coin_step:{coin_step}')
        print(
            f' full_df_size: {len(self.df)}, trade_range_size:{trade_range_size}, max_steps: {max_steps}({max_steps / len(self.df) * 100:.2f}%)')
        print(f' no_action_finish:{no_action_finish}')
        print(f' df_sample(last row): {self.df[-1, exclude_cols_left:]}')
        print(f' slippage statistics (avg, stddev): {slippage}')
        print(f' init_balance:{init_balance}, position_ratio:{position_ratio}')
        print(f' save_ratio:{save_ratio}, stop_loss:{stop_loss} take_profit:{take_profit}')
        if visualize:
            self.dates = date2num(df[:, 0].to_numpy())
            self.visualize = True
            self.render_range = render_range
            self.time_step = self.dates[1] - self.dates[0]
            print(f' Visualization enabled, time step: {self.time_step} (as factor of day)')
        else:
            self.visualize = False
            print(f' Visualize is set to false or there was no dates df provided.')
        if write_to_file:
            self.write_to_file = True
            self.filename = f'{REPORT_DIR}/envs/{self.__class__.__name__}_{str(dt.today()).replace(":", "-")[:-3]}.csv'
            with open(self.filename, 'w', newline='') as f:
                header = ['trade_id', 'trade_type', 'position_size', 'quantity', 'balance', 'save_balance',
                          'profit', 'fees', 'sl_losses']
                writer(f).writerow(header)
            self.to_file = []
        else:
            self.write_to_file = False
        self.verbose = verbose
        # This implementation uses only mean values provided by arg dict (slippage) #
        # as factor for calculation of real buy and sell prices. #
        # Generation of random numbers is too expensive computational wise. #
        # self.slippage = slippage
        if slippage is not None:
            self.buy_factor = slippage['buy'][0] + slippage['buy'][1] * slipp_std
            self.sell_factor = slippage['sell'][0] - slippage['sell'][1] * slipp_std
            self.stop_loss_factor = slippage['stop_loss'][0] - slippage['stop_loss'][1] * slipp_std
            # TODO: handle take profit slippage data
            self.take_profit_factor = 1.0
        else:
            self.buy_factor, self.sell_factor, self.stop_loss_factor, self.take_profit_factor = 1.0, 1.0, 1.0, 1.0
        self.save_ratio = save_ratio
        self.save_balance = 0.0
        self.total_steps = len(self.df)
        self.exclude_cols_left = exclude_cols_left
        self.no_action_finish = no_action_finish
        self.coin_step = coin_step
        self.fee = fee
        self.max_steps = max_steps
        self.init_balance = init_balance
        self.position_ratio = position_ratio
        self.position_size = self.init_balance * self.position_ratio
        self.balance = self.init_balance
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        # Discrete action space: 0 - hold, 1 - buy, 2 - sell
        self.action_space = spaces.Discrete(3)
        # Observation space #
        # none_df_obs_count are observations like current PnL, account balance, asset quantity etc. #
        # obs_space_dims = len(self.df[0, exclude_cols_left:])
        # obs_lower_bounds = array([-inf for _ in range(obs_space_dims)])
        # obs_upper_bounds = array([inf for _ in range(obs_space_dims)])
        # self.observation_space = spaces.Box(low=obs_lower_bounds, high=obs_upper_bounds)

    # Reset the state of the environment to an initial state
    def reset(self, **kwargs):
        self.creation_t = time()
        self.done = False
        self.reward = 0
        if self.visualize:
            self.visualization = TradingGraph(self.render_range, self.time_step)
        if self.write_to_file:
            self.filename = f'{REPORT_DIR}/envs/{self.__class__.__name__}_{str(dt.today()).replace(":", "-")[:-3]}.csv'
            with open(self.filename, 'w', newline='') as f:
                header = ['trade_id', 'trade_type', 'position_size', 'quantity', 'balance', 'save_balance',
                          'profit', 'fees', 'sl_losses']
                writer(f).writerow(header)
            self.to_file = []
        self.last_order_type = ''
        self.info = {}
        self.PLs_and_ratios = []
        self.balance = self.init_balance
        self.position_size = self.init_balance * self.position_ratio
        self.prev_bal = 0
        self.enter_price = 0
        self.stop_loss_price, self.take_profit_price = 0, 0
        self.qty = 0
        self.pnl = 0
        self.absolute_profit = 0.0
        self.SL_losses, self.cumulative_fees, self.liquidations, self.take_profits_c = 0, 0, 0, 0
        self.in_position, self.in_position_counter, self.position_closed = 0, 0, 0
        self.episode_orders, self.with_gain_c = 0, 1
        self.good_trades_count, self.bad_trades_count = 1, 1
        self.max_drawdown, self.max_profit = 0, 0
        self.loss_hold_counter, self.profit_hold_counter = 0, 0
        self.max_balance = self.min_balance = self.balance
        self.save_balance = 0.0
        if self.max_steps > 0:
            self.start_step = randint(self.start_index, self.end_index - self.max_steps)
            self.end_step = self.start_step + self.max_steps - 1
        else:
            self.start_step = self.start_index
            self.end_step = self.end_index
        self.current_step = self.start_step
        self.obs = iter(self.df[self.start_step:self.end_step, self.exclude_cols_left:])
        # return self.df[self.current_step, self.exclude_cols_left:]
        return next(self.obs)

    # def _next_observation(self):
    #   self.current_step += 1
    #   if self.current_step==self.end_step-1:
    #     self._finish_episode()
    #   return next(self.obs)
    #
    # def _random_factor(self, price, trade_type):
    #   return round(price*float(normalvariate(self.slippage[trade_type][0], self.slippage[trade_type][1])), 2)

    def _buy(self, price):
        if self.stop_loss is not None:
            self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
        if self.take_profit is not None:
            self.take_profit_price = round((1 + self.take_profit) * price, 2)
        # Considering random factor as in real world scenario #
        # price = self._random_factor(price, 'market_buy')
        adj_price = price * self.buy_factor
        # When there is no fee, subtract 1 just to be sure balance can buy this amount #
        step_adj_qty = floor(self.position_size / (adj_price * self.coin_step))
        if step_adj_qty == 0:
            self._finish_episode()
            return
        self.last_order_type = 'open_long'
        self.in_position = 1
        self.episode_orders += 1
        self.enter_price = adj_price
        self.qty = step_adj_qty * self.coin_step
        self.position_size = self.qty * adj_price
        self.prev_bal = self.balance
        self.balance -= self.position_size
        fee = (self.position_size * self.fee)
        self.position_size -= fee
        self.cumulative_fees += fee
        self.absolute_profit = -fee
        if self.write_to_file:
            self._write_to_file()
        # print(f'BOUGHT {self.qty} at {price}({adj_price})')

    def _sell(self, price, sl=False, tp=False):
        if sl:
            # price = self._random_factor(price, 'SL')
            # while price>self.enter_price:
            #     price = self._random_factor(price, 'SL')
            adj_price = price * self.stop_loss_factor
            if adj_price > self.enter_price:
                raise RuntimeError(
                    f"Stop loss price is above position enter price. (sl_factor={self.stop_loss_factor})")
            self.last_order_type = 'stop_loss_long'
        elif tp:
            self.take_profits_c += 1
            adj_price = price * self.take_profit_factor
            # TODO: add new order type for visualizations
            self.last_order_type = 'close_long'
        else:
            # price = self._random_factor(price, 'market_sell')
            adj_price = price * self.sell_factor
            self.last_order_type = 'close_long'
        _value = self.qty * adj_price
        self.balance += round(_value, 2)
        fee = _value * self.fee
        self.balance -= fee
        self.cumulative_fees += fee
        percentage_profit = (self.balance / self.prev_bal) - 1
        self.absolute_profit = self.balance - self.prev_bal
        # print(f'SOLD {self.qty} at {price}({adj_price}) profit ${self.balance-self.prev_bal:.2f}')
        # PROFIT #
        if percentage_profit > 0:
            if self.save_ratio is not None:
                save_amount = self.absolute_profit * self.save_ratio
                self.save_balance += save_amount
                self.balance -= save_amount
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
                self.SL_losses += self.absolute_profit
        self.PLs_and_ratios.append((percentage_profit, self.good_trades_count / self.bad_trades_count))
        self.position_size = (self.balance * self.position_ratio)
        # If balance minus position_size and fee is less or eq 0 #
        if self.position_size < (price * self.coin_step):
            self._finish_episode()
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.position_closed = 1
        self.stop_loss_price = 0
        if self.write_to_file:
            self._write_to_file()

    def _next_observation(self):
        try:
            self.current_step += 1
            return next(self.obs)
        except StopIteration:
            self.current_step -= 1
            self._finish_episode()
            return self.df[self.current_step, self.exclude_cols_left:]

    def step(self, action):
        self.last_order_type = ''
        self.absolute_profit = 0.0
        if self.in_position:
            high, low, close = self.df[self.current_step, 2:5]
            # print(f'low: {low}, close: {close}, self.enter_price: {self.enter_price}')
            self.in_position_counter += 1
            self.pnl = (close / self.enter_price) - 1
            if self.pnl >= 1:
                self.profit_hold_counter += 1
            else:
                self.loss_hold_counter += 1
            # Handling stop losses and take profits
            if (self.stop_loss is not None) and (low <= self.stop_loss_price):
                self._sell(self.stop_loss_price, sl=True)
            elif (self.take_profit is not None) and (high >= self.take_profit_price):
                self._sell(self.take_profit_price, tp=True)
            elif action == 2 and self.qty > 0:
                self._sell(close)
        elif action == 1:
            close = self.df[self.current_step, 4]
            self._buy(close)
        elif (not self.episode_orders) and ((self.current_step - self.start_step) > self.no_action_finish):
            self._finish_episode()
        else:
            if self.init_balance < self.balance + self.save_balance:
                self.with_gain_c += 1
        # Older version:
        # return self._next_observation(), self.reward, self.done, self.info
        return self._next_observation(), self.reward, self.done, False, self.info

    def _finish_episode(self):
        # print('BacktestEnv._finish_episode()')
        if self.in_position:
            self._sell(self.enter_price)
        self.done = True
        # Summary
        self.PNL_arrays = array(self.PLs_and_ratios)
        self.balance += self.save_balance
        gain = self.balance - self.init_balance
        total_return = (self.balance / self.init_balance) - 1
        risk_free_return = (self.df[self.end_step, 4] / self.df[self.start_step, 4]) - 1
        above_free = (total_return - risk_free_return)
        # if hasattr(self, 'leverage'):
        # above_free /= self.leverage
        hold_ratio = self.profit_hold_counter / self.loss_hold_counter if self.loss_hold_counter > 1 and self.profit_hold_counter > 1 else 1.0
        if len(self.PNL_arrays) > 1:
            mean_pnl, stddev_pnl = mean(self.PNL_arrays[:, 0]), std(self.PNL_arrays[:, 0])
            profits = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] > 0]
            losses = self.PNL_arrays[:, 0][self.PNL_arrays[:, 0] < 0]
            profits_mean = mean(profits) if len(profits) > 1 else 0.0
            losses_mean = mean(losses) if len(losses) > 1 else 0.0
            losses_stddev = std(losses) if len(losses) > 1 else 0.0
            PnL_trades_ratio = mean(self.PNL_arrays[:, 1])
            PnL_means_ratio = abs(profits_mean / losses_mean) if profits_mean * losses_mean != 0 else 1.0
            # slope_indicator = linear_slope_indicator(PnL_trades_ratio)
            slope_indicator = 1.000
            steps = self.max_steps if self.max_steps > 0 else self.total_steps
            in_gain_indicator = self.with_gain_c / (
                    steps - self.profit_hold_counter - self.loss_hold_counter - self.episode_orders)
            if above_free > 0:
                if hasattr(self, 'leverage'):
                    above_free_factor = above_free
                    # above_free_factor = above_free / self.leverage**(1/3)
                    # above_free_factor = above_free/sqrt(self.leverage)
                else:
                    above_free_factor = above_free
                self.reward = (above_free_factor * self.episode_orders * PnL_trades_ratio * (
                        hold_ratio ** (1 / 3)) * (PnL_means_ratio ** (1 / 3)) * in_gain_indicator) / steps
            else:
                self.reward = (above_free * self.episode_orders * 1 / PnL_trades_ratio * 1 / (
                        hold_ratio ** (1 / 3)) * 1 / (PnL_means_ratio ** (1 / 3)) * 1 / in_gain_indicator) / steps
            # self.reward = total_return*100
        else:
            mean_pnl, stddev_pnl = 0.0, 0.0
            profits_mean, losses_mean, losses_stddev = 0.0, 0.0, 0.0
            PnL_trades_ratio, PnL_means_ratio = 0.0, 0.0
            in_gain_indicator = 0.0
            slope_indicator = 0.000
            self.reward = -inf

        sharpe_ratio = (mean_pnl - risk_free_return) / stddev_pnl if stddev_pnl != 0 else -1
        sortino_ratio = (total_return - risk_free_return) / losses_stddev if losses_stddev != 0 else -1
        # with_gain_c is not incremented when in position and while position is being opened, so we need to subtract those values from 'total_steps
        # sl_losses_adj_gain = gain-self.SL_losses
        # self.reward = copysign((abs(gain)**1.5)*self.PL_count_mean*sqrt(hold_ratio)*sqrt(self.PL_ratio)*sqrt(self.episode_orders), gain)/self.total_steps
        # self.reward = copysign(gain**2, gain)+(self.episode_orders/sqrt(self.total_steps))+self.PL_count_mean+sqrt(hold_ratio)+sqrt(self.PL_ratio)
        exec_time = time() - self.creation_t
        if self.balance >= 1_000_000:
            self.verbose = True
        if self.verbose:
            print(
                f'Episode finished: gain:${gain:.2f}({total_return * 100:.2f}%), gain/step:${gain / (self.end_step - self.start_step):.5f}, ',
                end='')
            print(
                f'cumulative_fees:${self.cumulative_fees:.2f}, SL_losses:${self.SL_losses:.2f} take_profits:{self.take_profits_c}')
            print(f' save_ratio:{self.save_ratio}, saved_balance:${self.save_balance:.2f}')
            print(
                f' trades:{self.episode_orders:_}, trades_with(profit/loss):{self.good_trades_count - 1:_}/{self.bad_trades_count - 1:_}, ',
                end='')
            print(f'trades_avg(profit/loss):{profits_mean * 100:.2f}%/{losses_mean * 100:.2f}%, ', end='')
            print(f'max(profit/drawdown):{self.max_profit * 100:.2f}%/{self.max_drawdown * 100:.2f}%,')
            print(f' PnL_trades_ratio:{PnL_trades_ratio:.3f}, PnL_means_ratio:{PnL_means_ratio:.3f}, ', end='')
            print(
                f'hold_ratio:{hold_ratio:.3f}, PNL_mean:{mean_pnl * 100:.2f}%, ', end='')
            print(
                f'geo_avg_return:{((self.balance / self.init_balance) ** (1 / (self.end_step - self.start_step)) - 1) * 100:.7f}%')
            print(
                f' slope_indicator:{slope_indicator:.4f}, in_gain_indicator:{in_gain_indicator:.3f}, sharpe_ratio:{sharpe_ratio:.2f}, sortino_ratio:{sortino_ratio:.2f},',
                end='')
            print(f' risk_free:{risk_free_return * 100:.2f}%, above_free:{above_free * 100:.2f}%,')
            print(f' reward:{self.reward:.8f} exec_time:{exec_time:.2f}s')
        if self.write_to_file:
            with open(self.filename, 'a', newline='') as f:
                for row in self.to_file:
                    writer(f).writerow(row)
        self.info = {'gain': gain,
                     'PnL_means_ratio': PnL_means_ratio,
                     'PnL_trades_ratio': PnL_trades_ratio,
                     'hold_ratio': hold_ratio,
                     'PNL_mean': mean_pnl,
                     'slope_indicator': slope_indicator,
                     'exec_time': exec_time}

    def render(self, indicator_or_reward=None, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            if self.in_position:
                _pnl = self.df[self.current_step, 4] / self.enter_price - 1
                _balance = self.balance + (self.position_size + (self.position_size * _pnl))
            else:
                _balance = self.balance
            if indicator_or_reward is None:
                indicator_or_reward = self.df[self.current_step, -1]
            render_row = [self.dates[self.current_step],
                          *self.df[self.current_step, 1:5],
                          indicator_or_reward,
                          _balance]
            trade_info = [self.last_order_type, str(round(self.absolute_profit, 2))]
            #   print(f'trade_info {trade_info}')
            self.visualization.append(render_row, trade_info)
            self.visualization.render()

    def _write_to_file(self):
        _row = [self.episode_orders, self.last_order_type, round(self.position_size, 2),
                round(self.qty, 8), round(self.balance, 2), round(self.save_balance, 2), round(self.absolute_profit, 2),
                round(self.cumulative_fees, 2), round(self.SL_losses, 2)]
        self.to_file.append(_row)


class FuturesBacktest(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # https://www.binance.com/en/futures/trading-rules/perpetual/leverage-margin
        self.POSITION_TIER = {1: (125, .0040, 0), 2: (100, .005, 50),
                              3: (50, .01, 2_550), 4: (20, .025, 122_550),
                              5: (10, .05, 1_372_550), 6: (5, .10, 5_372_550),
                              7: (4, .125, 7_872_550), 8: (3, .15, 10_872_550),
                              9: (2, .25, 30_872_550), 10: (1, .50, 105_872_550)}
        if 'leverage' in kwargs:
            self.leverage = kwargs['leverage']
        else:
            self.leverage = 1
        self.df_mark = kwargs['df_mark'].to_numpy()
        # BTCUSDTperp last 1Y mean=6.09e-05 stdev=6.52e-05, mean+2*stedv covers ~95,4% of variance
        # self.funding_rate = 0.01912 * (1/100)
        self.funding_rate = 0.01 * (1 / 100)
        print(
            f' df_mark_size: {len(self.df_mark)}, max_steps: {self.max_steps}({self.max_steps / len(self.df) * 100:.2f}%)')

    def reset(self, **kwargs):
        self.margin = 0
        self.liquidation_price = 0
        self.tier = 0
        # self.stop_loss /= self.leverage
        self.liquidations = 0
        self.liquidation_losses = 0
        return super().reset(**kwargs)

    def _check_tier(self):
        # print('_check_tier')
        if self.position_size < 50_000:
            self.tier = 1
        elif 50_000 < self.position_size < 500_000:
            self.tier = 2
        elif 500_000 < self.position_size < 8_000_000:
            self.tier = 3
        elif 8_000_000 < self.position_size < 50_000_000:
            self.tier = 4
        elif 50_000_000 < self.position_size < 80_000_000:
            self.tier = 5
        elif 80_000_000 < self.position_size < 100_000_000:
            self.tier = 6
        elif 100_000_000 < self.position_size < 120_000_000:
            self.tier = 7
        elif 120_000_000 < self.position_size < 200_000_000:
            self.tier = 8
        elif 200_000_000 < self.position_size < 300_000_000:
            self.tier = 9
        elif 300_000_000 < self.position_size < 500_000_000:
            self.tier = 10
        if self.leverage > self.POSITION_TIER[self.tier][0]:
            # print(f' Leverage exceeds tier {self.tier} max', end=' ')
            # print(f'changing from {self.leverage} to {self.POSITION_TIER[self.tier][0]} (Balance:${self.balance}:.2f)')
            self.leverage = self.POSITION_TIER[self.tier][0]

    # def _check_margin(self):
    #   #print('_check_margin')
    #   if self.qty>0:
    #     min_price = self.df[self.current_step, 2]
    #   elif self.qty<0:
    #     min_price = self.df[self.current_step, 1]
    #   else:
    #     pass
    #     #print('co Ty tu robisz?')
    #   position_value = abs(self.qty*min_price)
    #   unrealized_PNL = abs(self.qty*self.enter_price/self.leverage)*self._get_pnl(min_price)
    #   # 1.25% Liquidation Clearance
    #   margin_balance = self.margin + unrealized_PNL - (position_value*0.0125) - (position_value*self.fee)
    #   maintenance_margin = position_value*self.POSITION_TIER[self.tier][1]-self.POSITION_TIER[self.tier][2]
    #   print(f'min_price:{min_price:.2f} position_value:{position_value:.2f} unrealized_PNL:{unrealized_PNL:.2f} Clearance:{(position_value*0.0125)} fee:{(position_value*self.fee)} margin:{self.margin} margin_balance:{margin_balance:.2f} maintenance_margin:{maintenance_margin:.2f} margin_ratio:{maintenance_margin/margin_balance*100}')
    #   if maintenance_margin>margin_balance:
    #     return True
    #   else:
    #     return False
    def _check_margin(self):
        # If in long position and mark Low below liquidation price
        if self.qty > 0:
            return self.liquidation_price >= self.df_mark[self.current_step, 3]
        # If in short position and mark High above liquidation price
        elif self.qty < 0:
            return self.liquidation_price <= self.df_mark[self.current_step, 2]
        return False

    def _get_pnl(self, price, update=False):
        if update and self.in_position:
            self.pnl = ((price / self.enter_price) - 1) * self.sign_leverage
            self.loss_hold_counter += (self.pnl < 0)
            self.profit_hold_counter += (self.pnl > 0)
            return self.pnl
        elif not self.in_position:
            return 0
        return ((price / self.enter_price) - 1) * self.sign_leverage

    def _open_position(self, side, price):
        if side == 'long':
            adj_price = price * self.buy_factor
            if self.stop_loss is not None:
                self.stop_loss_price = round((1 - self.stop_loss) * price, 2)
            if self.take_profit is not None:
                self.take_profit_price = round((1 + self.take_profit) * price, 2)
            self.last_order_type = 'open_long'
        elif side == 'short':
            adj_price = price * self.sell_factor
            if self.stop_loss is not None:
                self.stop_loss_price = round((1 + self.stop_loss) * price, 2)
            if self.take_profit is not None:
                self.take_profit_price = round((1 - self.take_profit) * price, 2)
            self.last_order_type = 'open_short'
        else:
            raise RuntimeError('side should be "long" or "short"')
        self._check_tier()
        adj_qty = floor(self.position_size * self.leverage / (adj_price * self.coin_step))
        if adj_qty == 0:
            adj_qty = 1
            # print('Forcing adj_qty to 1. Calculated quantity possible to buy with given postion_size and coin_step equals 0')
        self.margin = (adj_qty * self.coin_step * adj_price) / self.leverage
        if self.margin > self.balance:
            self._finish_episode()
            return
        self.prev_bal = self.balance
        self.balance -= self.margin
        fee = (self.margin * self.fee * self.leverage)
        self.margin -= fee
        self.cumulative_fees -= fee
        self.absolute_profit = -fee
        self.in_position = 1
        self.episode_orders += 1
        self.enter_price = price
        if side == 'long':
            self.qty = adj_qty * self.coin_step
        elif side == 'short':
            self.qty = -1 * adj_qty * self.coin_step
        # for speeding up _get_pnl() method
        self.sign_leverage = copysign(1, self.qty) * self.leverage
        # https://www.binance.com/en/support/faq/how-to-calculate-liquidation-price-of-usd%E2%93%A2-m-futures-contracts-b3c689c1f50a44cabb3a84e663b81d93
        # 1,25% liquidation clearance fee https://www.binance.com/en/futures/trading-rules/perpetual/
        self.liquidation_price = (self.margin * (1 - 0.0125) - self.qty * self.enter_price) / (
                abs(self.qty) * self.POSITION_TIER[self.tier][1] - self.qty)
        # print(f'OPENED {side} price:{price} adj_price:{adj_price} qty:{self.qty} margin:{self.margin} fee:{fee}')
        # sleep(10)
        if self.write_to_file:
            self._write_to_file()

    def _close_position(self, price, liquidated=False, sl=False, tp=False):
        if sl:
            adj_price = price * self.stop_loss_factor
            if self.qty > 0:
                self.last_order_type = 'stop_loss_long'
            elif self.qty < 0:
                self.last_order_type = 'stop_loss_short'
        if tp:
            self.take_profits_c += 1
            adj_price = price * self.take_profit_factor
            # TODO: add new order type for visualizations
            if self.qty > 0:
                self.last_order_type = 'take_profit_long'
            elif self.qty < 0:
                self.last_order_type = 'take_profit_short'
        else:
            if self.qty > 0:
                adj_price = price * self.sell_factor
                self.last_order_type = 'close_long'
            elif self.qty < 0:
                adj_price = price * self.buy_factor
                self.last_order_type = 'close_short'
            else:
                raise RuntimeError("Bad call to _close_position, qty is 0")
        _position_value = abs(self.qty) * adj_price
        _fee = (_position_value * self.fee)
        if liquidated:
            if self.qty > 0:
                self.last_order_type = 'liquidate_long'
            elif self.qty < 0:
                self.last_order_type = 'liquidate_short'
            margin_balance = 0
            self.liquidation_losses -= self.margin
        else:
            unrealized_PNL = (abs(self.qty) * self.enter_price / self.leverage) * self._get_pnl(adj_price)
            margin_balance = self.margin + unrealized_PNL - _fee
        self.cumulative_fees -= _fee
        self.balance += margin_balance
        self.margin = 0
        percentage_profit = (self.balance / self.prev_bal) - 1
        self.absolute_profit = self.balance - self.prev_bal
        ### PROFIT
        if percentage_profit > 0:
            if self.save_ratio is not None:
                save_amount = self.absolute_profit * self.save_ratio
                self.save_balance += save_amount
                self.balance -= save_amount
            self.good_trades_count += 1
            if self.balance >= self.max_balance:
                self.max_balance = self.balance
            if (self.max_profit == 0) or (percentage_profit > self.max_profit):
                self.max_profit = percentage_profit
        ### LOSS
        elif percentage_profit < 0:
            self.bad_trades_count += 1
            if self.balance <= self.min_balance:
                self.min_balance = self.balance
            if (self.max_drawdown == 0) or (percentage_profit < self.max_drawdown):
                self.max_drawdown = percentage_profit
            if sl:
                self.SL_losses += self.absolute_profit
        self.PLs_and_ratios.append((percentage_profit, self.good_trades_count / self.bad_trades_count))
        self.position_size = (self.balance * self.position_ratio)
        self.qty = 0
        self.in_position = 0
        self.in_position_counter = 0
        self.position_closed = 1
        self.stop_loss_price = 0
        self.pnl = 0
        # print(f'CLOSED {self.last_order_type} price:{price} adj_price:{adj_price} qty:{self.qty} percentage_profit:{percentage_profit} absolute_profit:{self.absolute_profit} margin:{self.margin} fee:{_fee}')
        # sleep(10)
        if self.write_to_file:
            self._write_to_file()

    # Execute one time step within the environment
    def step(self, action):
        self.last_order_type = ''
        self.absolute_profit = 0.0
        if self.in_position:
            high, low, close = self.df[self.current_step, 2:5]
            mark_close = self.df_mark[self.current_step, 4]
            self.in_position_counter += 1
            # change value for once in 8h, for 1m TF 8h=480
            if self.in_position_counter % 480 == 0:
                self.margin -= (abs(self.qty) * close * self.funding_rate)
            self._get_pnl(mark_close, update=True)
            if self._check_margin():
                self.liquidations += 1
                self._close_position(mark_close, liquidated=True)
            elif (self.stop_loss is not None) and (((low <= self.stop_loss_price) and (self.qty > 0)) or (
                    (high >= self.stop_loss_price) and (self.qty < 0))):
                self._close_position(self.stop_loss_price, sl=True)
            elif (self.take_profit is not None) and (((high >= self.take_profit_price) and (self.qty > 0)) or (
                    (low <= self.take_profit_price) and (self.qty < 0))):
                self._close_position(self.take_profit_price, tp=True)
            elif (action == 1 and self.qty < 0) or (action == 2 and self.qty > 0):
                self._close_position(close)
        elif action == 1:
            self._open_position('long', self.df[self.current_step, 4])
        elif action == 2:
            self._open_position('short', self.df[self.current_step, 4])
        else:
            self.with_gain_c += (self.init_balance < self.balance + self.save_balance)
        # self.info = {'action': action,
        #         'reward': 0,
        #         'step': self.current_step,
        #         'exec_time': time() - self.creation_t}
        return self._next_observation(), self.reward, self.done, False, self.info

    def _finish_episode(self):
        if self.in_position:
            self._close_position(self.enter_price)
        super()._finish_episode()
        if self.verbose:
            print(f' liquidations: {self.liquidations} liq_losses: ${self.liquidation_losses:.2f}')

    def render(self, indicator_or_reward=None, visualize=False, *args, **kwargs):
        if visualize or self.visualize:
            if self.in_position:
                _balance = self.balance + (self.margin + (self.margin * self.pnl))
            else:
                _balance = self.balance
            if indicator_or_reward is None:
                indicator_or_reward = self.df[self.current_step, -1]
            render_row = [self.dates[self.current_step],
                          *self.df[self.current_step, 1:5],
                          indicator_or_reward,
                          _balance]
            trade_info = [self.last_order_type, round(self.absolute_profit, 2)]
            #   print(f'trade_info {trade_info}')
            self.visualization.append(render_row, trade_info)
            self.visualization.render()
