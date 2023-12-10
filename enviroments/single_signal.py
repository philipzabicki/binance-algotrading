from enviroments.backtest import SpotBacktest
from numpy import hstack, empty


class SignalExecuteSpotEnv(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'enter_at' in kwargs:
            self.enter_threshold = kwargs['enter_at']
            self.close_threshold = kwargs['close_at']
        else:
            self.enter_threshold = 1.0
            self.close_threshold = 1.0
        self.df = hstack((self.df, empty((self.df.shape[0], 1))))

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
        obs = args[0]
        while not self.done:
            signal = obs[-1]
            action = 0
            if self.qty == 0:
                if signal >= self.enter_threshold:
                    action = 1
                elif signal <= -self.enter_threshold:
                    action = 2
            elif (self.qty < 0) and (signal >= self.close_threshold):
                action = 1
            elif (self.qty > 0) and (signal <= -self.close_threshold):
                action = 2
            obs, _, _, _, _ = self.step(action)
            if self.visualize:
                # current_step manipulation just to synchronize plot rendering
                # could be fixed by calling .render() inside .step() just before return statement
                self.current_step -= 1
                self.render()
                self.current_step += 1
        return obs, self.reward, self.done, False, self.info
