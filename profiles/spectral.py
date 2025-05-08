"""
Spectral analysis of neuronal responses.

## Functions

- `whitenoise()`: band-limited white noise.
- `rate()`: firing rate computed by kernel convolution.
- `spectra()`: stimulus- and response power spectra, and cross spectrum.
- `susceptibilities()`: stimulus- and response spectra up to second order.
- `diag_projection()`: projection of the chi2 matrix onto its diagonal.
- `hor_projection()`: horizontal projection of the chi2 matrix.
- `peakedness()`: normalized size of a peak expected around a specific frequency.

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


def spectra(stimulus, spikes, dt, nfft):
    """Stimulus- and response power spectra, and cross spectrum.

    Compute the complex-valued transfer function (first-order
    susceptibility) and the stimulus-response coherence like this:

    ```
    freqs, pss, prr, prs = spectra(stimulus, spikes, dt, nfft)
    transfer = prs/pss
    coherence = np.abs(prs)**2/pss/prr
    ```

    The gain of the transfer function is the absolute value of the
    transfer function:

    ```
    gain = np.abs(prs)/pss
    ```

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
    prs: ndarray of complex
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


def susceptibilities(stimulus, spikes, dt=0.0005, nfft=2**9):
    """ Stimulus- and response spectra up to second order.

    Compute the complex-valued transfer function (first-order
    susceptibility) and the stimulus-response coherence like this:

    ```
    freqs, pss, prr, prs, prss, n = susceptibilities(stimulus, spikes, dt, nfft)
    transfer = prs/pss
    coherence = np.abs(prs)**2/pss/prr
    ```

    The gain of the transfer function is the absolute value of the
    transfer function:

    ```
    gain = np.abs(prs)/pss
    ```

    The second-order susceptibility can be computed like this:

    ```
    chi2 = prss*0.5/np.sqrt(pss.reshape(1, -1)*pss.reshape(-1, 1))
    ```

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
        Power spectrum of the response averaged over segments.
    prs: ndarray of complex
        Cross spectrum between stimulus and response averaged over segments.
    prss: ndarray of complex
        Cross bispectrum between stimulus and response averaged over segments.
    n: int
        Number of segments.
    """
    freqs = np.fft.fftfreq(nfft, dt)
    idx = np.argmin(freqs)
    freqs = np.roll(freqs, -idx)
    f0 = np.argmin(np.abs(freqs))   # index of zero frequency
    fidx = np.arange(len(freqs))
    fsum_idx = fidx.reshape(-1, 1) + fidx.reshape(1, -1) - f0
    fsum_idx[fsum_idx < 0] = 0
    fsum_idx[fsum_idx >= len(fidx)] = len(fidx) - 1
    f0 = len(freqs)//4
    f1 = 3*len(freqs)//4
    segments = range(0, len(stimulus) - nfft, nfft)
    # stimulus:
    p_ss = np.zeros(len(freqs))
    fourier_s = np.zeros((len(segments), len(freqs)), complex)
    n = 0
    for j, k in enumerate(segments):
        fourier_s[j] = np.fft.fft(stimulus[k:k + nfft], n=nfft)
        fourier_s[j] = np.roll(fourier_s[j], -idx)
        p_ss += np.abs(fourier_s[j]*np.conj(fourier_s[j]))
        n += 1
    p_ss /= n
    # response spectra:
    time = np.arange(len(stimulus))*dt
    p_rr = np.zeros(len(freqs))
    p_rs = np.zeros(len(freqs), complex)
    p_rss = np.zeros((len(freqs), len(freqs)), complex)
    n = 0
    for i, spiket in enumerate(spikes):
        b, _ = np.histogram(spiket, time)
        b = b / dt
        for j, k in enumerate(segments):
            # stimulus:
            fourier_s1 = fourier_s[j].reshape(len(fourier_s[j]), 1)
            fourier_s2 = fourier_s[j].reshape(1, len(fourier_s[j]))
            fourier_s12 = np.conj(fourier_s1)*np.conj(fourier_s2)
            # response:
            fourier_r = np.fft.fft(b[k:k + nfft] - np.mean(b), n=nfft)
            fourier_r = np.roll(fourier_r, -idx)
            p_rr += np.abs(fourier_r*np.conj(fourier_r))
            p_rs += np.conj(fourier_s[j])*fourier_r
            p_rss += fourier_s12*fourier_r[fsum_idx]
            n += 1
    return freqs[f0:f1], p_ss[f0:f1], p_rr[f0:f1]/n, p_rs[f0:f1]/n, p_rss[f0:f1, f0:f1]/n, n


def diag_projection(freqs, chi2, fmax):
    """ Projection of the chi2 matrix onto its diagonal.

    Adapted from https://stackoverflow.com/questions/71362928/average-values-over-all-offset-diagonals

    Parameters
    ----------
    freqs: ndarray of float
        Frequencies of the chi2 matrix.
    chi2: 2-D ndarray of float
        Second-order susceptibility matrix.
    fmax: float
        Maximum frequency for the projection.

    Returns
    -------
    dfreqs: ndarray of float
        Frequencies of the projection.
    diagp: ndarray of float
        Projections of the chi2 matrix onto its diagonal.
        That is, averages over the anti-diagonals.
    """
    i0 = np.argmin(freqs < 0)
    i1 = np.argmax(freqs > fmax)
    if i1 == 0:
        i1 = len(freqs)
    chi2 = chi2[i0:i1, i0:i1]
    n = chi2.shape[0]
    diagp = np.zeros(n*2-1, dtype=float)
    for i in range(n):
        diagp[i:i + n] += chi2[i]
    diagp[0:n] /= np.arange(1, n+1, 1, dtype=float)
    diagp[n:]  /= np.arange(n-1, 0, -1, dtype=float)
    dfreqs = np.arange(len(diagp))*(freqs[i0 + 1] - freqs[i0]) + freqs[i0]
    return dfreqs, diagp


def hor_projection(freqs, chi2, fmax):
    """ Horizontal projection of the chi2 matrix.

    Parameters
    ----------
    freqs: ndarray of float
        Frequencies of the chi2 matrix.
    chi2: 2-D ndarray of float
        Second-order susceptibility matrix.
    fmax: float
        Maximum frequency for the projection.

    Returns
    -------
    hfreqs: ndarray of float
        Frequencies of the projection.
    horp: ndarray of float
        Projections of the chi2 matrix onto its x-axis.
        That is, averages over columns.
    """
    i0 = np.argmin(freqs < 0)
    i1 = np.argmax(freqs > fmax)
    if i1 == 0:
        i1 = len(freqs)
    hfreqs = freqs[i0:i1]
    chi2 = chi2[i0:i1, i0:i1]
    horp = np.mean(chi2, 1)
    return hfreqs, horp


def peakedness(freqs, projection, fbase, median=True,
               searchwin=40, averagewin=10):
    """Normalized size of a peak expected around a specific frequency.

    Parameters
    ----------
    freqs: ndarray of float
        Frequencies of the projection.
    projection: ndarray of float
        Projection of the chi2 matrix.
    fbase: float
        The neurons baseline-frequency.
        That is, the frequency where a peak is expected in the projection.
    median: bool
        If True, normalize the peak height by the median of the projection.
        Otherwise (default), normalize by averaged values of the projection
        close to the fbase frequency.
    searchwin: float
        Search for peak in the projection at fbase plus and
        minus `searchwin` Hertz.
    averagewin: float
        For estimating the reference level, an average is taken in two
        `averagewin` Hertz wide windows that are located `averagewin`
        Hertz to the left and right of the detected peak.

    Returns
    -------
    p: float
        The normalized height of the peak close to fbase.
    fpeak: float
        The frequency of the detected peak.

    """
    sel = (freqs > fbase - searchwin) & (freqs < fbase + searchwin)
    snippet = projection[sel]
    if len(snippet) == 0:
        return np.nan, np.nan
    peak = np.max(snippet)
    fpeak = freqs[np.argmax(snippet) + np.argmax(sel)]
    bleft = np.nan
    bright = np.nan
    if median:
        baseline = np.median(projection)
    else:
        mask = (freqs >= fpeak - 2*averagewin) & (freqs <= fpeak - averagewin)
        bleft = np.mean(projection[mask]) if np.sum(mask) > 0 else np.nan
        mask = (freqs >= fpeak + averagewin) & (freqs <= fpeak + 2*averagewin)
        bright = np.mean(projection[mask]) if np.sum(mask) > 0 else np.nan
        if np.isfinite(bleft) and np.isfinite(bright):
            baseline = 0.5*(bleft + bright)
        elif np.isfinite(bleft):
            baseline = bleft
        elif np.isfinite(bright):
            baseline = bright
        else:
            baseline = np.nan
    if np.isnan(peak/baseline):
        print(peak, fpeak, fbase, baseline, bleft, bright, median)
        return np.nan, np.nan
    return peak/baseline, fpeak

