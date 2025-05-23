""" 
EOD analysis

## Functions

- `eod_times()`: times of EOD zero crossings.
- `detect_eods()`: detect EOD times.
- `plot_eod_interval_hist()`: plot inter-EOD-interval histogram.

"""

import numpy as np
from thunderlab.eventdetection import detect_peaks, std_threshold


def eod_times(data, rate, eodf=None):
    """Times of EOD zero crossings.

    First, estimate EOD frequency from power spectrum.
    The band-pass filter the data around that frequency.
    From the filtered signal, detect and interpolate zero crossings.

    Parameters
    ----------
    data: ndarray of float
        Time series of an EOD.
    rate: float
        Sampling rate of the data.
    eodf: None or float
        An estimate of the EOD frequency. If provided, the EOD
        frequency is not estimated from the power spectrum of the
        data.

    Returns
    -------
    times: ndarray of float
        Times of EOD zero crossings.

    """
    # estimate EOD frequency:
    if eodf is None:
        nsamples = 0.2*rate  # samples that make up 0.2s (5Hz resolution)
        nfft = 2**8
        while nfft < nsamples:
            nfft *= 2
        freq, power = welch(data - np.mean(data), rate,
                            nperseg=nfft, noverlap=nfft//2)
        eodf = freq[np.argmax(power)]
    # band-pass filter:
    sos = butter(2, [0.5*eodf, 1.5*eodf], 'bandpass', fs=rate, output='sos')
    fdata = sosfiltfilt(sos, data)
    # detect zero crossings:
    idx = np.nonzero((fdata[:-1] <= 0) & (fdata[1:] > 0))[0]
    # interpolate:
    time = np.arange(len(fdata))/rate
    eodtimes = np.zeros(len(idx))
    for k in range(len(idx)):
        i = idx[k]
        eodtimes[k] = np.interp(0, fdata[i:i + 2], time[i:i + 2])
    return eodtimes


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
