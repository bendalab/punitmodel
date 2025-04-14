"""
Analyse stationary spike trains

## Functions

- `interval_statistics()`: statistics and kde of interspike intervals.
- `serial_correlations()`: serial correlations of interspike intervals.
- `vector_strength()`: vector strength of spike times relative to a periodic signal.
- `cyclic_rate()`: kernel density estimate of spike times relative to a periodic signal.

"""

import numpy as np
from scipy.stats import gaussian_kde


def interval_statistics(spikes, sigma=1e4, maxisi=0.1):
    """ Statistics and kde of interspike intervals.

    Parameters
    ----------
    spikes: nparray of floats
        Spike times of baseline activity.
    sigma: float
        Standard deviation of Gaussian kernel used for for kde
        of interspike intervals. Same unit as `spikes`.
    maxisi: float or None
        Maximum interspike interval for kde. If None or 0, use maximum interval.

    Returns
    -------
    isis: ndarray of floats
        Interspike intervals for kde.
    kde: ndarray of floats
        Kernel density estimate of interspike intervals for each `isis`.
        Plot it like this:
        ```
        ax.fill_between(isis, kde)
        ```
    rate: float
        Mean baseline firing rate as inverse mean interspike interval.
    cv: float
        Coefficient of variation (std divided by mean) of the interspike intervals.
    """
    intervals = np.diff(spikes)
    if not maxisi:
        maxisi = np.max(intervals) + 3*sigma
    isis = np.arange(0.0, maxisi, 0.1*sigma)
    kernel = gaussian_kde(intervals, sigma/np.std(intervals, ddof=1))
    kde = kernel(isis)
    mean_isi = np.mean(intervals)
    std_isi = np.std(intervals)
    rate = 1/mean_isi
    cv = std_isi / mean_isi
    return isis, kde, rate, cv


def serial_correlations(spikes, max_lag=10):
    """ Serial correlations of interspike intervals.

    Parameters
    ----------
    spikes: nparray of floats
        Spike times of baseline activity.
    max_lag: int
        Compute serial correlations up to this lag.

    Returns
    -------
    lags: ndarray of ints
        Lags for which interspike interval correlations have been computed.
        First one is zero, last one is `max_lag`.
    corrs: ndarray of floats
        Serial correlations for all `lags`.
    low: float
        0.1% percentile of the null hypothesis of no correlation.
    high: float
        99.9% percentile of the null hypothesis of no correlation.
    """
    intervals = np.diff(spikes)
    lags = np.arange(0, max_lag + 1, 1)
    corrs = np.zeros(max_lag + 1)
    corrs[0] = np.corrcoef(intervals, intervals)[0,1]
    for i, lag in enumerate(lags[1:]):
        corrs[i+1] = np.corrcoef(intervals[:-lag], intervals[lag:])[0,1]
    # permuation test:
    rng = np.random.default_rng()
    perm_corrs = np.zeros(10000)
    for k in range(len(perm_corrs)):
        xintervals = rng.permutation(intervals)
        yintervals = rng.permutation(intervals)
        perm_corrs[k] = np.corrcoef(xintervals, yintervals)[0,1]
    low, high = np.quantile(perm_corrs, (0.001, 0.999))
    return lags, corrs, low, high


def vector_strength(spikes, cycles):
    r""" Vector strength of spike times relative to a periodic signal.

    Computes for each spike time its phase $\varphi_i$
    within the period of `cycles`.
    Vector strength is then
    $$vs = \left|\frac[1}{n} \sum_{i=1}^n e^{i\varphi_i} \right|$$

    Parameters
    ----------
    spikes: nparray of floats
        Spike times.
    cycles: float or nparray of floats
        Times of full periods. A single number indicates the period
        of the periodic signal.

    Returns
    -------
    vs: float
        Computed vector strength.
    """
    if np.isscalar(cycles):
        cycles = np.arange(0, spikes[-1] + 10*cycles, cycles)    
    vectors = np.zeros(len(spikes), dtype=complex)
    for i, spike in enumerate(spikes):
        k = cycles.searchsorted(spike) - 1
        if k + 1 >= len(cycles):
            vectors = vectors[:i]
            break
        cycle = cycles[k]
        period = cycles[k+1] - cycles[k]
        phase = 2*np.pi*(spike - cycle)/period
        vectors[i] = np.exp(1j*phase)
    vs = np.abs(np.mean(vectors))
    return vs
        

def cyclic_rate(spikes, cycles, sigma=0.05):
    """Kernel density estimate of spike times relative to a periodic signal.

    Computes for each spike time its phase $\varphi_i$ within the
    period of `cycles`. Kernel density estimate is then from these
    phases.

    Parameters
    ----------
    spikes: nparray of floats
        Spike times.
    cycles: float or nparray of floats
        Times of full periods. A single number indicates the period
        of the periodic signal.
    sigma: float
        Standard deviation of Gaussian kernel used for for kde
        of interspike intervals. Between 0 and 1, 1 corresponds to a full period.

    Returns
    -------
    phases: ndarray of floats
        Phases at which the kde is computed.
        Step size is set to a tenth of `sigma`.
    kde: ndarray of floats
        The kernel density estimate of the spike times within periods of `cycles`.
    """
    if np.isscalar(cycles):
        cycles = np.arange(0, spikes[-1] + 10*cycles, cycles)    
    phases = np.arange(0, 1.005, 0.1*sigma)*2*np.pi
    rate = np.zeros(len(phases))
    n = 0
    for i, spike in enumerate(spikes):
        k = cycles.searchsorted(spike) - 1
        if k + 1 >= len(cycles):
            break
        cycle = cycles[k]
        period = cycles[k+1] - cycles[k]
        phase = 2*np.pi*(spike - cycle)/period
        cycle_spikes = np.array([phase - 2*np.pi, phase, phase + 2*np.pi])
        kernel = gaussian_kde(cycle_spikes, 2*np.pi*sigma/np.std(cycle_spikes, ddof=1))
        cycle_rate = kernel(phases)
        rate += cycle_rate
        n += 1
    return phases, rate/n
        
