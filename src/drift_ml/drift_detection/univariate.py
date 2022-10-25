from scipy import stats
import typing
import collections
import random
import itertools
import numpy as np

# from .common import DriftDetector


class SimpleSTD:
    """Reference data based simple standard deviation drift detector

    Parameters
    ----------
    std_thresh
        Threshold for drift detection. A drift will be detected if the mean
        of the window values is
        >= mean(reference_window) (+/-) std_thresh * std(reference_window)
    """

    def __init__(
        self,
        std_thresh: float = 3,
        window_size: int = 30,
        shift_reference_window: bool = False,
        window: typing.Iterable = None,
    ) -> None:
        self.std_thresh = std_thresh
        self.window_size = window_size
        self.shift_reference_window = shift_reference_window

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    @property
    def drift_detected(self):
        return self._drift_detected

    def _reset(self) -> None:
        self._drift_detected = False
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)

    def fit(self, reference_data) -> None:
        self.reference_window = reference_data
        self.reference_std = np.std(reference_data)
        self.reference_mean = np.mean(reference_data)
        self.upper_limit = self.reference_mean + (self.std_thresh * self.reference_std)
        self.lower_limit = self.reference_mean - (self.std_thresh * self.reference_std)

    def update(self, x) -> None:
        if self._drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:

            window_mean = np.mean(self.window)
            if window_mean >= self.upper_limit or window_mean <= self.lower_limit:
                self._drift_detected = True
                if self.shift_reference_window:
                    self.fit(self.window)
            else:
                self._drift_detected = False
            return window_mean
        else:
            self._drift_detected = False
            return None


class KSReference:
    """Reference data based Kolmogorov-Smirnov drift detector

    Parameters
    ----------
    alpha
        Probability for the test statistic of the Kolmogorov-Smirnov-Test.
        The alpha parameter is very sensitive, therefore should be set below 0.01.
    window_size
        Size of the sliding window.
    shift_reference_window
        If true, the reference window will be changed to the current window
        if a drift is detected
    window
        Already collected data to avoid cold start.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 30,
        shift_reference_window=False,
        window: typing.Iterable = None,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.shift_reference_window = shift_reference_window
        self.fitted = False

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    @property
    def drift_detected(self):
        return self._drift_detected

    def _reset(self) -> None:
        self._drift_detected = False
        self.p_value = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)

    def fit(self, reference_data) -> None:
        self.reference_window = reference_data
        self.fitted = True

    def update(self, x) -> None:
        if not self.fitted:
            raise Exception

        if self._drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:

            st, self.p_value = stats.ks_2samp(self.window, self.reference_window)

            if self.p_value <= self.alpha and st > 0.1:
                self._drift_detected = True
                if self.shift_reference_window:
                    self.fit(self.window)
            else:
                self._drift_detected = False
        else:
            self._drift_detected = False


class KSWIN:
    """Sliding window based Kolmogorov-Smirnov drift detector

    https://github.com/online-ml/river/blob/main/river/drift/kswin.py

    Parameters
    ----------
    alpha
        Probability for the test statistic of the Kolmogorov-Smirnov-Test.
        The alpha parameter is very sensitive, therefore should be set below 0.01.
    window_size
        Size of the sliding window.
    stat_size
        Size of the statistic window.
    window
        Already collected data to avoid cold start.
    seed
        Random seed for reproducibility.

    References
    ----------
    [^1]: Christoph Raab, Moritz Heusinger, Frank-Michael Schleif,
    Reactive Soft Prototype Computing for
    Concept Drift Streams, Neurocomputing, 2020,
    """

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        window: typing.Iterable = None,
        seed=None,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

        self._reset()

        if window:
            self.window = collections.deque(window, maxlen=self.window_size)

    @property
    def drift_detected(self):
        return self._drift_detected

    def _reset(self) -> None:
        self._drift_detected = False
        self.p_value = 0
        self.n = 0
        self.window: typing.Deque = collections.deque(maxlen=self.window_size)
        self._rng = random.Random(self.seed)

    def update(self, x) -> None:
        if self._drift_detected:
            self._reset()

        self.n += 1

        self.window.append(x)
        if len(self.window) >= self.window_size:
            rnd_window = [
                self.window[r]
                for r in self._rng.sample(
                    range(self.window_size - self.stat_size), self.stat_size
                )
            ]
            most_recent = list(
                itertools.islice(
                    self.window, self.window_size - self.stat_size, self.window_size
                )
            )

            st, self.p_value = stats.ks_2samp(rnd_window, most_recent)

            if self.p_value <= self.alpha and st > 0.1:
                self._drift_detected = True
                self.window = collections.deque(most_recent, maxlen=self.window_size)
            else:
                self._drift_detected = False
        else:
            self._drift_detected = False
