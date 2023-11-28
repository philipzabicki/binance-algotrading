from enviroments.backtest import SpotBacktest


class SignalExecuteEnv(SpotBacktest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'close_at' in kwargs and 'enter_at' in kwargs:
            self.enter_threshold = kwargs['enter_at']
            self.close_threshold = kwargs['close_at']
        else:
            self.enter_threshold = 1.0
            self.close_threshold = 1.0

    def reset(self, *args, **kwargs):
        self.position_ratio = kwargs['position_ratio'] if 'position_ratio' in kwargs else 1.00
        self.stop_loss = kwargs['stop_loss'] if 'stop_loss' in kwargs else None
        self.enter_threshold = kwargs['enter_at'] if 'enter_at' in kwargs else 1.0
        self.close_threshold = kwargs['close_at'] if 'close_at' in kwargs else 1.0
        return super().reset()

    def _finish_episode(self):
        super()._finish_episode()
        if self.verbose:
            print(f' position_ratio={self.position_ratio}, stop_loss={self.stop_loss}', end='')
            print(f' enter_at={self.enter_threshold:.3f}, close_at={self.close_threshold:.3f}')

    def __call__(self, *args, **kwargs):
        obs = args[0]
        while not self.done:
            # obs[-1] assumes that signal values are stored in last column
            signal = obs[-1]
            # print(f'signal {signal} enter {self.enter_threshold} close {self.close_threshold}')
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
            self.current_step -= 1
            obs, _, _, _, _ = self.step(action)
            if self.visualize:
                self.render()
            self.current_step += 1
            print(f'obs: {obs} action: {action} qty: {self.qty}')
        return obs, self.reward, self.done, False, self.info
