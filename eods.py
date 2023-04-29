""" 
EOD analysis

## Functions

- `detect_eods()`: detect EOD times.
- `plot_eod_interval_hist()`: plot inter-EOD-interval histogram.

"""

import numpy as np
from thunderfish.eventdetection import detect_peaks, std_threshold


def detect_eods(data, samplerate):
    """Detect EOD times.
    
    Detect the peaks in `data` and compute the position of the vertex
    of the parabula given by the three data points around each peak.

    Parameters
    ----------
    data: ndarray of floats
        A recording trace of an EOD.
    samplerate: float
        Sampling rate of the recording in Hertz.

    Returns
    -------
    eod_times: ndarray of floats
        Timepoints of each EOD cycle.

    """
    thresh = std_threshold(data, thresh_fac=2)
    p, _ = detect_peaks(data, thresh)
    eod_times = p/samplerate
    # correct to peak of the vertex of a parabula through the 3 highest points:
    eod_times += (data[p+1] - data[p-1])/(2*data[p] - data[p-1] - data[p+1])/samplerate
    return eod_times


def plot_eod_interval_hist(ax, eod_times, max_iei=None):
    """ Plot inter-EOD-interval histogram.

    Parameters
    ----------
    ax: matplotlib axes
        Axes for plotting the histogram.
    eod_times: ndarray of floats
        Times od the EODs in seconds.
    max_iei: None or float
        Maximum inter-EOD interval in milliseconds.
        If None, make histogram just around the data.
    """
    ieis = np.diff(eod_times)
    maxiei = 1000*np.max(ieis)
    ax.axvline(maxiei, color='gray', ls=':')
    if max_iei:
        ax.hist(1000*ieis, np.arange(0, max_iei, 0.1))
    else:
        ax.hist(1000*ieis, len(ieis)//50)
    ax.set_ylim(0.8, None)
    ax.set_yscale('log')
