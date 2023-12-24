from numpy import empty

from .base import SpotBacktest, FuturesBacktest


class SignalExecuteSpotEnv(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'enter_at' in kwargs:
            self.enter_threshold = kwargs['enter_at']
            self.close_threshold = kwargs['close_at']
        else:
            self.enter_threshold = 1.0
            self.close_threshold = 1.0
        self.signals = empty(self.total_steps)

    def reset(self, *args, **kwargs):
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.00
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.enter_threshold = kwargs['enter_at'] if 'enter_at' in kwargs else 1.0
        self.close_threshold = kwargs['close_at'] if 'close_at' in kwargs else 1.0
        return super().reset()

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' position_ratio={self.position_ratio:.2f}, stop_loss={self.stop_loss * 100:.3f}%,', end='')
            print(f' enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}')

    def __call__(self, *args, **kwargs):
        while not self.done:
            # step must be start_step adjusted cause one can start and end backtest at any point in df
            _step = self.current_step - self.start_step
            if self.signals[_step] >= self.close_threshold:
                action = 1
            elif self.signals[_step] <= -self.close_threshold:
                action = 2
            else:
                action = 0
            _, _, _, _, _ = self.step(action)
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render(indicator_or_reward=self.signals[_step])
                self.current_step += 1
        return None, self.reward, self.done, False, self.info


class SignalExecuteFuturesEnv(FuturesBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'enter_at' in kwargs and 'leverage' in kwargs:
            self.enter_threshold = kwargs['enter_at']
            self.close_threshold = kwargs['close_at']
            self.leverage = kwargs['leverage']
        else:
            self.enter_threshold = 1.0
            self.close_threshold = 1.0
            self.leverage = 5
        self.signals = empty(self.total_steps)

    def reset(self, *args, **kwargs):
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.0
        self.leverage = kwargs['leverage'] if 'leverage' in kwargs else 5
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.enter_threshold = kwargs['enter_at'] if 'enter_at' in kwargs else 1.0
        self.close_threshold = kwargs['close_at'] if 'close_at' in kwargs else 1.0
        return super().reset()

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(
                f' position_ratio={self.position_ratio:.2f}, leverage={self.leverage}, stop_loss={self.stop_loss * 100:.3f}%,',
                end='')
            print(f' enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}')

    def __call__(self, *args, **kwargs):
        while not self.done:
            # step must be start_step adjusted cause one can start and end backtest at any point in df
            _step = self.current_step - self.start_step
            if self.signals[_step] >= self.close_threshold:
                action = 1
            elif self.signals[_step] <= -self.close_threshold:
                action = 2
            else:
                action = 0
            _, _, _, _, _ = self.step(action)
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render(indicator_or_reward=self.signals[_step])
                self.current_step += 1
        return None, self.reward, self.done, False, self.info
