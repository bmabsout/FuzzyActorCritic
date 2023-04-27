import signal
import scipy.signal
import time
import numpy as np
from typing import Callable, Union

start_time = 0
i = 0
def loop_every(interval: float, callback: Callable[[int, float], None]):
    """Call the callback every interval seconds.
    The callback will be called with two arguments:
    1. The number of times the callback has been called.
    2. The time elsapsed since the first call.
    and the time since the first call.
    """
    def handler(signum, frame):
        global i, start_time
        if i == 0:
            start_time = time.time()
        elapsed_time = time.time() - start_time
        callback(i, elapsed_time)
        i += 1
    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, interval, interval)


def noise( t: Union[float, np.ndarray], min_freq: float = 30.0, max_freq: float = 50.0, num_waves: int = 30, seed=42):
    """ Generate a random noise signal, and evaluate it at time t in seconds. t can be a float or a numpy array
        The noise signal is a sum of sinusoids with random frequencies, amplitudes and phases.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(min_freq, max_freq, size=num_waves)
    amplitudes = rng.uniform(0.0, 0.01, size=num_waves)
    phases = rng.uniform(0, 2*np.pi, size=num_waves)
    angles = np.multiply.outer(freqs*2*np.pi, t).T + phases
    return np.sum(amplitudes*np.sin(angles), axis=-1)


class Stateful():
    """A class representing which turn a function of state and input into a stateful function of just inputs, taking care of the plumbing.
        Example usage:
        >>> def stateless_fun(state, val):
        ...     new_state = state + val
        ...     return new_state, new_state
        >>> stateful = Stateful(0, stateful_fun)
        >>> stateful.step(1)
        1
        >>> stateful.step(2)
        3
    """
    def __init__(self, init_state, stateful_fun):
        self.state = init_state
        self.stateful_fun = stateful_fun

    def step(self, val):
        new_val, self.state = self.stateful_fun(self.state, val)
        return new_val

def windowed(squash_window, window_size=5) -> Stateful:
    """
        returns a stateful function which has a rolling buffer and applies a function to the buffer to produce a value.
    """
    def windowed_step(window, new_val):
        window.append(new_val)
        if len(window) > window_size:
            window = window[1:]
        filtered_val = squash_window(np.array(window).flatten()) if len(window) == window_size else new_val
        return filtered_val, window
    return Stateful([], windowed_step)

def Median_filter():
    return windowed(np.median, window_size=5)


def Butterworth_filter(butterworth_order=3, cutoff_freq=30, sample_freq=100):
    b, a = scipy.signal.butter(butterworth_order, cutoff_freq / (sample_freq / 2.0), analog=False)
    def butterworth_step(state, val):
        new_val, new_state = scipy.signal.lfilter(b, a, [val], zi=state)
        return new_val[0], new_state
    return Stateful(scipy.signal.lfilter_zi(b, a), butterworth_step)
