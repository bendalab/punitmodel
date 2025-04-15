"""
Spectral analysis of neuronal responses.

## Functions


"""

import numpy as np
from scipy.signal import welch, csd
from scipy.stats import norm


def whitenoise(cflow, cfup, dt, duration, rng=np.random.default_rng()):
    """Band-limited white noise.

    Generates white noise with a flat power spectrum between `cflow` and
    `cfup` Hertz, zero mean and unit standard deviation.  Note, that in
    particular for short segments of the generated noise the mean and
    standard deviation of the returned noise can deviate from zero and
    one.

    Parameters
    ----------
    cflow: float
        Lower cutoff frequency in Hertz.
    cfup: float
        Upper cutoff frequency in Hertz.
    dt: float
        Time step of the resulting array in seconds.
    duration: float
        Total duration of the resulting array in seconds.

    Returns
    -------
    noise: 1-D array
        White noise.
    """
    # number of elements needed for the noise stimulus:
    n = int(np.ceil((duration+0.5*dt)/dt))
    # next power of two:
    nn = int(2**(np.ceil(np.log2(n))))
    # indices of frequencies with `cflow` and `cfup`:
    inx0 = int(np.round(dt*nn*cflow))
    inx1 = int(np.round(dt*nn*cfup))
    if inx0 < 0:
        inx0 = 0
    if inx1 >= nn/2:
        inx1 = nn/2
    # draw random numbers in Fourier domain:
    whitef = np.zeros((nn//2+1), dtype=complex)
    # zero and nyquist frequency must be real:
    if inx0 == 0:
        whitef[0] = 0
        inx0 = 1
    if inx1 >= nn//2:
        whitef[nn//2] = 1
        inx1 = nn//2-1
    phases = 2*np.pi*rng.random(size=inx1 - inx0 + 1)
    whitef[inx0:inx1+1] = np.cos(phases) + 1j*np.sin(phases)
    # inverse FFT:
    noise = np.real(np.fft.irfft(whitef))
    # scaling factor to ensure standard deviation of one:
    sigma = nn / np.sqrt(2*float(inx1 - inx0))
    return noise[:n]*sigma


def spectra(stimulus, spikes, dt, nfft):
    time = np.arange(len(stimulus))*dt
    freq, pss = welch(stimulus, fs=1/dt, nperseg=nfft, noverlap=nfft//2)
    prr = np.zeros((len(spikes), len(freq)))
    prs = np.zeros((len(spikes), len(freq)), dtype=complex)
    for i, spiket in enumerate(spikes):
        b, _ = np.histogram(spiket, time)
        b = b / dt
        f, rr = welch(b - np.mean(b), fs=1/dt, nperseg=nfft, noverlap=nfft//2)
        f, rs = csd(b - np.mean(b), stimulus,
                    fs=1/dt, nperseg=nfft, noverlap=nfft//2)
        prr[i] = rr
        prs[i] = rs
    return freq, pss, np.mean(prr, 0), np.mean(prs, 0)


def rate(time, spikes, sigma):
    kernel = norm.pdf(time[time < 8*sigma], loc=4*sigma, scale=sigma)
    rates = np.zeros((len(spikes), len(time)))
    xtime = np.append(time, time[-1] + time[1] - time[0])
    for i, spiket in enumerate(spikes):
        b, _ = np.histogram(spiket, xtime)
        rates[i] = np.convolve(b, kernel, 'same')
    return np.mean(rates, 0), np.std(rates, 0)

