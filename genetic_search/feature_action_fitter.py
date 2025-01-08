from numpy import sqrt, mean, where, float64, corrcoef, nan_to_num, inf, array
from pandas import to_numeric
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer, Choice
from talib import AD
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean, hamming
from sklearn.metrics import mean_squared_error
from scipy.signal import correlate
# from tslearn.metrics import dtw
from utils.ta_tools import get_ma_band_signal_by_source, get_MA_band_signal, custom_MACD, MACD_cross_signal, custom_StochasticOscillator, StochasticOscillator_signal, custom_ChaikinOscillator, ChaikinOscillator_signal, custom_MACD_with_source

class KeltnerChannelFitting(ElementwiseProblem):
    def __init__(self, df, *args, **kwargs):
        self.weighted_actions = df['Action']*df['Weight']
        print(f'df {df}')
        self.df = df.to_numpy()
        bands_variables = {"atr_period": Integer(bounds=(2, 500)),
                           "ma_type": Integer(bounds=(0, 0)),
                           "ma_period": Integer(bounds=(2, 500)),
                           "atr_multi": Real(bounds=(0.001, 15.000))}
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X: {X}')
        signals = nan_to_num(get_MA_band_signal(self.df[:, 1:6].astype(float),
                                     X['ma_type'], X['ma_period'],
                                     X['atr_period'], X['atr_multi']))
        # print(f'self.weighted_actions {self.weighted_actions}')
        # print(f'signals {signals}')
        signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = 1 - spearmanr(signals, self.weighted_actions)[0]
        # r, _ = spearmanr(signals, self.weighted_actions)
        # score = 1 - r
        out["F"] = euclidean(signals*self.df[:,-1], self.weighted_actions)
        # out["F"] = sqrt(mean((signals - self.weighted_actions) ** 2))


class KeltnerChannelVarSourceFitting(ElementwiseProblem):
    def __init__(self, df, lower, upper, ma_type, *args, **kwargs):
        self.ma_type = ma_type
        self.lower, self.upper = lower, upper
        self.actions = df['Action'].values
        self.weights = df['Weight'].values
        self.df = df.to_numpy()
        bands_variables = {
            "atr_period": Integer(bounds=(2, 1000)),
            "ma_period": Integer(bounds=(2, 1000)),
            "atr_multi": Real(bounds=(0.001, 15.000)),
            "source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"])
        }
        super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        signals = get_ma_band_signal_by_source(
            self.df[:, 1:6].astype(float),
            self.ma_type,
            X['ma_period'],
            X['atr_period'],
            X['atr_multi'],
            X['source']
        )
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))

        mask = (self.weights > self.lower) & (self.weights <= self.upper)
        signals_masked = signals[mask]
        actions_masked = self.actions[mask]
        # out["F"] = sqrt(mean((nan_to_num(signals_masked) - actions_masked) ** 2))
        # out["F"] = euclidean(nan_to_num(signals_masked), actions_masked)
        out["F"] = mean_squared_error(nan_to_num(signals_masked), actions_masked)

# class KeltnerChannelVarSourceFitting(ElementwiseProblem):
#     def __init__(self, df, lower, upper, ma_type, *args, **kwargs):
#         self.ma_type = ma_type
#         df.loc[abs(df['Weight']) < lower, ['Weight', 'Action']] = [0.0, 0]
#         df.loc[abs(df['Weight']) > upper, ['Weight', 'Action']] = [0.0, 0]
#         self.actions = df['Action']
#         self.weights = df['Weight']
#         self.weighted_actions = df['Action']*df['Weight']
#         self.df = df.to_numpy()
#         bands_variables = {"atr_period": Integer(bounds=(2, 1_000)),
#                            # "ma_type": Integer(bounds=(0, 31)),
#                            "ma_period": Integer(bounds=(2, 1_000)),
#                            "atr_multi": Real(bounds=(0.001, 15.000)),
#                            "source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"])
#                            }
#         super().__init__(*args, vars=bands_variables, n_obj=1, **kwargs)
#
#     def _evaluate(self, X, out, *args, **kwargs):
#         # print(f'X: {X}')
#         signals = get_ma_band_signal_by_source(self.df[:, 1:6].astype(float),
#                                      self.ma_type, X['ma_period'],
#                                      X['atr_period'], X['atr_multi'],
#                                                X['source'])
#         signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
#         print(signals)
#         # out["F"] = 1 - spearmanr(signals, self.weighted_actions)[0]
#         # out["F"] = sqrt(mean((signals*self.weights - self.weighted_actions) ** 2))
#         out["F"] = euclidean(nan_to_num(signals) * self.weights, self.weighted_actions)


class MACDVarSourceFitting(ElementwiseProblem):
    def __init__(self, df, lower, upper, *args, **kwargs):
        self.lower, self.upper = lower, upper
        self.actions = df['Action'].values
        self.weights = df['Weight'].values
        self.df = df.to_numpy()
        macd_variables = {"fast_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
                          "slow_source": Choice(options=["open", "high", "low", "close", "hl2", "hlc3", "ohlc4", "hlcc4"]),
                          "fast_period": Integer(bounds=(2, 1000)),
                          "slow_period": Integer(bounds=(2, 1000)),
                          "signal_period": Integer(bounds=(2, 1000)),
                          "fast_ma_type": Integer(bounds=(0, 31)),
                          "slow_ma_type": Integer(bounds=(0, 31)),
                          "signal_ma_type": Integer(bounds=(0, 25))}
        super().__init__(*args, vars=macd_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        macd, macd_signal = custom_MACD_with_source(self.df[:, 1:6].astype(float),
                                                    fast_source=X['fast_source'],
                                                    slow_source=X['slow_source'],
                                                    fast_ma_type=X['fast_ma_type'], fast_period=X['fast_period'],
                                                    slow_ma_type=X['slow_ma_type'], slow_period=X['slow_period'],
                                                    signal_ma_type=X['signal_ma_type'], signal_period=X['signal_period'])
        signals = array(MACD_cross_signal(macd, macd_signal))
        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        mask = (self.weights > self.lower) & (self.weights <= self.upper)
        # print(f'mask {mask}')
        signals_masked = signals[mask]
        actions_masked = self.actions[mask]

        # out["F"] = sqrt(mean((signals - self.weighted_actions) ** 2))
        # out["F"] = euclidean(nan_to_num(signals_masked), actions_masked)
        out["F"] = mean_squared_error(nan_to_num(signals_masked), actions_masked)


class StochSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, *args, **kwargs):
        self.weighted_actions = df['Action'] * df['Weight']
        self.df = df.to_numpy()
        stoch_variables = {"oversold_threshold": Real(bounds=(0.0, 50.0)),
                          "overbought_threshold": Real(bounds=(50.0, 100.0)),
                          "fastK_period": Integer(bounds=(2, 250)),
                          "slowK_period": Integer(bounds=(2, 250)),
                          "slowD_period": Integer(bounds=(2, 250)),
                          "slowK_ma_type": Integer(bounds=(0, 25)),
                          "slowD_ma_type": Integer(bounds=(0, 25))
                           }
        super().__init__(*args, vars=stoch_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        slowK, slowD = custom_StochasticOscillator(self.df[:, 1:6],
                                                   fastK_period=X['fastK_period'],
                                                   slowK_period=X['slowK_period'],
                                                   slowD_period=X['slowD_period'],
                                                   slowK_ma_type=X['slowK_ma_type'],
                                                   slowD_ma_type=X['slowD_ma_type'])
        signals = StochasticOscillator_signal(slowK,
                                               slowD,
                                               oversold_threshold=X['oversold_threshold'],
                                               overbought_threshold=X['overbought_threshold'])

        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = sqrt(mean((signals - self.weighted_actions) ** 2))
        out["F"] = euclidean(nan_to_num(signals), self.weighted_actions)


class ChaikinOscillatorSpotMixedVariableProblem(ElementwiseProblem):
    def __init__(self, df, *args, **kwargs):
        # Zakładam, że df jest obiektem pandas DataFrame
        # Najpierw upewnij się, że odpowiednie kolumny są typu float
        # Zastąp 'High', 'Low', 'Close', 'Volume' rzeczywistymi nazwami kolumn
        columns_to_convert = ['High', 'Low', 'Close', 'Volume']

        for col in columns_to_convert:
            df[col] = to_numeric(df[col], errors='coerce')

        # Zamiast usuwać wiersze, możemy wypełnić brakujące wartości
        # Na przykład, używając wypełnienia metodą 'ffill' (forward fill)
        df[columns_to_convert] = df[columns_to_convert].fillna(method='ffill')

        # Alternatywnie, można użyć interpolacji:
        # df[columns_to_convert] = df[columns_to_convert].interpolate(method='linear')

        # Teraz przekonwertuj kolumny na float64
        high = df['High'].astype(float64).values
        low = df['Low'].astype(float64).values
        close = df['Close'].astype(float64).values
        volume = df['Volume'].astype(float64).values

        # Przekonwertuj cały DataFrame na NumPy, jeśli jest to potrzebne
        self.df = df.to_numpy()
        print(f'self.df  {self.df[0, :]}')  # Sprawdzenie pierwszego wiersza

        # Oblicz ADL
        self.adl = AD(high, low, close, volume)

        self.weighted_actions = df['Action'] * df['Weight']
        chaikin_variables = {"fast_period": Integer(bounds=(2, 500)),
                             "slow_period": Integer(bounds=(2, 500)),
                             "fast_ma_type": Integer(bounds=(0, 25)),
                             "slow_ma_type": Integer(bounds=(0, 25))}
        super().__init__(*args, vars=chaikin_variables, n_obj=1, **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # print(f'X {X}')
        chaikin_oscillator = custom_ChaikinOscillator(self.adl,
                                                      fast_ma_type=X['fast_ma_type'], fast_period=X['fast_period'],
                                                      slow_ma_type=X['slow_ma_type'], slow_period=X['slow_period'])
        signals = ChaikinOscillator_signal(chaikin_oscillator)

        # signals = where(signals >= 1, 1, where(signals <= -1, -1, 0))
        # out["F"] = sqrt(mean((signals - self.weighted_actions) ** 2))
        out["F"] = euclidean(nan_to_num(signals), self.weighted_actions)