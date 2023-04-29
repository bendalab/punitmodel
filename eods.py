""" 
EOD analysis

## Functions

- `plot_eod_interval_hist()`: plot inter-EOD-interval histogram.

"""

import numpy as np


def plot_eod_interval_hist(ax, eod_times, max_iei=5):
    """ Plot inter-EOD-interval histogram.

    Parameters
    ----------
    ax: matplotlib axes
        Axes for plotting the histogram.
    eod_times: ndarray of floats
        Times od the EODs in seconds.
    max_iei: float
        Maximum inter-EOD interval in milliseconds.
    """
    ieis = np.diff(eod_times)
    maxiei = 1000*np.max(ieis)
    ax.axvline(maxiei, color='gray', ls=':')
    ax.hist(1000*ieis, np.arange(0, max_iei, 0.1))
    ax.set_ylim(0.8, None)
    ax.set_yscale('log')
