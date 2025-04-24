"""
Spectral analysis of neuronal responses.

## Functions

- `whitenoise()`: band-limited white noise.
- `spectra()`: stimulus- and response power spectra, and cross spectrum.
- `rate()`: firing rate computed by kernel convolution.

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
    """ Stimulus- and response power spectra, and cross spectrum.

    Parameters
    ----------
    stimulus: ndarray of float
        Stimulus waveform with sampling interval 'dt'.
    spikes: list of ndarrays of float
        Spike times in response to the stimulus.
    dt: float
        Sampling interval of stimulus and resolution of the binary spike train.
    nfft: int
        Number of samples used for each Fourier transformation.

    Returns
    -------
    freqs: ndarray of float
        The frequencies corresponding to the spectra.
    pss: ndarray of float
        Power spectrum of the stimulus.
    prr: ndarray of float
        Power spectrum of the response averaged over trials.
    psr: ndarray of complex
        Cross spectrum between stimulus and response averaged over trials.
    """
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
    """ Firing rate computed by kernel convolution.

    Parameters
    ----------
    time: ndarray of float
        Times at which firing rate is evaluated.
    spikes: list of ndarray of float
        Spike times.
    sigma: float
       Width of the Gaussian kernel as a standard deviation.

    Returns
    -------
    rate: ndarray of float
        Firing rate of convolved spike trains averaged over trials.
    ratesd: ndarray of float
        Corresponding standard deviation.
    """
    kernel = norm.pdf(time[time < 8*sigma], loc=4*sigma, scale=sigma)
    rates = np.zeros((len(spikes), len(time)))
    xtime = np.append(time, time[-1] + time[1] - time[0])
    for i, spiket in enumerate(spikes):
        b, _ = np.histogram(spiket, xtime)
        rates[i] = np.convolve(b, kernel, 'same')
    return np.mean(rates, 0), np.std(rates, 0)

