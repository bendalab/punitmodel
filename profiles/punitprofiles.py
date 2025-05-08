import sys
sys.path.insert(0, '..')  # for model.py
import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
from model import simulate, load_models
from eods import plot_eod_interval_hist
from baseline import interval_statistics, burst_fraction, serial_correlations
from baseline import vector_strength, cyclic_rate
from spectral import whitenoise, spectra, rate
import plottools.plottools as pt


def plot_style():
    """ Settings and styles for the plots.

    Returns
    -------
    s: namespace
        Plotting styles (dictionaries with colors, line poperties, ...).
    """
    class s:
        colors = pt.palettes['muted']
        data_color = colors['blue']
        model_color = colors['red']
        onset_color = colors['green']
        ss_color = colors['red']
        base_color = colors['lightblue']
        error_color = colors['red']
        poscontrast_color = colors['magenta']
        negcontrast_color = colors['purple']
        stimulus_color = colors['darkgreen']
        rate_color = colors['red']
        
        lw = 1.5
        lwthin = 0.7
        ms = 3
        
        lsGrid = dict(color='gray', ls=':', lw=lwthin)
        lsData = dict(color=data_color, lw=lwthin)
        lsModel = dict(color=model_color, lw=lwthin)
        lpsData = dict(color=data_color, ls='-', lw=lw, marker='o', ms=ms, clip_on=False)
        lpsModel = dict(color=model_color, ls='-', lw=lw, marker='o', ms=ms, clip_on=False)
        fsData = dict(facecolor=data_color, edgecolor='none', alpha=0.5)
        fsModel = dict(facecolor=model_color, edgecolor='none', alpha=0.5)
        psOnset = dict(ls='none', color=onset_color, marker='o', ms=ms, clip_on=False)
        psSS = dict(ls='none', color=ss_color, marker='o', ms=ms, clip_on=False)
        lsOnset = dict(ls='-', color=onset_color, lw=lw)
        lsSS = dict(ls='-', color=ss_color, lw=lw)
        lsBase = dict(ls='-', color=base_color, lw=lw)
        lsDataLine = dict(color='black', lw=lw, zorder=-10)
        lsModelLine = dict(color='gray', lw=lw, zorder=-10)
        lsRaster = dict(color='k', linewidths=0.5, lineoffsets=1, linelengths=0.7)
        lsStim = dict(color=stimulus_color, lw=lw)
        lsRate = dict(color=rate_color, lw=lw)
        fsRate = dict(facecolor=rate_color, edgecolor='none', alpha=0.5)
        lsSpec = dict(color=model_color, lw=lw)

        lsPosContrast = dict(ls='-', color=poscontrast_color, lw=lw)
        lsNegContrast = dict(ls='-', color=negcontrast_color, lw=lw)
        fsError = dict(color=error_color, alpha=1)
    pt.axes_params(0, 0, 0, 'none')
    pt.figure_params(format='pdf')
    pt.spines_params('lb')
    pt.text_params(8)
    pt.legend_params(fontsize='small', frameon=False,
                     handlelength=1.2, borderpad=0)
    return s

    
def instantaneous_rate(spikes, time):
    """Firing rate as the inverse of the interspike intervals.

    Parameter
    ---------
    spikes: ndarrays of floats
        Spike times of a single trial.
    time: ndarray of floats
        Times on which instantaneous rate is computed.

    Returns
    -------
    rate: ndarray of floats
        Instantaneous firing rate corresponding to `spikes`.
    """
    isis = np.diff(spikes)                        # well, the ISIs
    inst_rate = 1 / isis                          # rate as inverse ISIs
    # indices (int!) of spike times in time array:
    dt = time[1] - time[0]
    spike_indices = np.asarray(np.round((spikes-time[0])/dt), int)
    spike_indices = spike_indices[(spike_indices >= 0) &
                                  (spike_indices < len(time))]
    rate = np.zeros(len(time))
    for i in range(len(spike_indices)-1): # for all spikes and ISIs, except the last
        # set the full ISI to the instantaneous rate of that ISI:
        rate[spike_indices[i]:spike_indices[i+1]] = inst_rate[i]
    return rate


def plot_comparison(ax, s, title, error):
    ax.show_spines('r')
    ax.tick_params(axis='both', which='major', labelsize='x-small')
    ax.text(0, 1.1, title, transform=ax.transAxes, ha='left', fontsize='small')
    if error < 0.02:
        error = 0.02
    ax.bar(0, 100*np.abs(error), width=1, **s.fsError)
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(0.0, 30)
    ax.set_yticks_fixed([0, 20], ['0%', '20%'])
    plt.setp(ax.get_yticklabels(), rotation=90, va='center')

    
def baseline_data(data_path, cell):
    """ Load baseline spike and EOD events measured in a cell.

    Parameters
    ----------
    data_path: str
         Base path with the cell data folders.
    cell: str
         Name of the cell its data folder.

    Returns
    -------
    spikes: ndarray of floats
        Times of spikes.
    eods: ndarray of floats
        Times of EOD cacles.
    """
    spikes = np.load(os.path.join(data_path, cell, 'baseline_spikes_trial_1.npy'))
    eods = np.load(os.path.join(data_path, cell, 'baseline_eods_trial_1.npy'))
    return spikes, eods


def baseline_model(EODf, model_params, tmax=10):
    """ Simulate baseline activity.

    Parameters
    ----------
    EODf: float
        EOD frequency in Hertz
    model_params: dict
        Model parameters.
    tmax: float
        Time to simulate baseline activity.
    """
    deltat = model_params['deltat']
    time = np.arange(0, tmax, deltat)
    # baseline EOD with amplitude 1:
    stimulus = np.sin(2*np.pi*EODf*time)
    # integrate the model:
    spikes = simulate(stimulus, **model_params)
    return spikes


def analyse_baseline(EODf, spikes, eods, data={}, max_eods=15.5,
                     max_lag=15):
    """ Compute baseline statistics. """
    eod_period = 1/EODf
    if isinstance(eods, (float, int)):
        duration = eods
        eods = eod_period
    else:
        duration = eods[-1] - eods[0]
    isis, kde, rate, cv = interval_statistics(spikes,
                                             sigma=0.05*eod_period,
                                             maxisi=max_eods*eod_period)
    lags, corrs, corrs_null = serial_correlations(spikes, max_lag=max_lag)
    vs = vector_strength(spikes, eods)
    cphase, crate = cyclic_rate(spikes, eods, sigma=0.02)
    bf, bft, thresh = burst_fraction(spikes, EODf)
    
    data['isis'] = isis
    data['hist'] = kde
    data['eodf/Hz'] = EODf
    data['nspikesbase'] = len(spikes)
    data['durationbase/s'] = duration
    data['ratebase/Hz'] = rate
    data['cvbase'] = cv
    data['lags'] = lags
    data['serialcorrs'] = corrs
    data['serialcorrnull'] = corrs_null
    data['vectorstrength'] = vs
    data['cyclic_phases'] = cphase
    data['cyclic_rates'] = crate
    data['burstfrac'] = bf
    data['burstfracthresh'] = bft
    data['burstthresh/s'] = thresh
    return data

    
def plot_baseline(axi, axc, axv, axb, s, EODf, data, model):
    """ Compute and plot baseline statistics for data and model spikes. """
    eod_period = 1/EODf
    # plot isih statistics:
    for eod in np.arange(1, int(data['isis'][-1]/eod_period), 1):
        axi.axvline(1000*eod*eod_period, **s.lsGrid)
    axi.fill_between(1000*data['isis'], data['hist']/1000, **s.fsData)
    axi.fill_between(1000*model['isis'], model['hist']/1000, **s.fsModel)
    axi.plot(1000*data['isis'], data['hist']/1000, **s.lsData)
    axi.plot(1000*model['isis'], model['hist']/1000, **s.lsModel)
    axi.text(1, 0.95, f'$r={data["ratebase/Hz"]:.0f}$Hz',
            transform=axi.transAxes, ha='right')
    axi.text(1, 0.83, f'$CV^{{(d)}}={data["cvbase"]:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.text(1, 0.71, f'$CV^{{(m)}}={model["cvbase"]:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.set_xlabel('ISI', 'ms')
    axi.set_ylabel('pdf', '1/ms')
    axi.set_xticks_delta(10)
    # plot serial correlations:
    axc.axhspan(-data['serialcorrnull'], data['serialcorrnull'], **s.fsData)
    axc.axhline(0, **s.lsGrid)
    axc.plot(data['lags'], data['serialcorrs'], label='data', **s.lpsData)
    axc.plot(model['lags'], model['serialcorrs'], label='model', **s.lpsModel)
    axc.text(1, 0.95, f'$\\rho^{{(d)}}_1={data["serialcorrs"][1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.text(1, 0.83, f'$\\rho^{{(m)}}_1={model["serialcorrs"][1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.set_ylim(-1, 1)
    axc.set_xlabel('Lag $k$')
    axc.set_ylabel('Correlation $\\rho_k$')
    axc.set_xticks_delta(5)
    axc.legend(loc='lower right', markerfirst=False)
    # plot vector strength:
    axv.fill_between(data['cyclic_phases'], data['cyclic_rates'], **s.fsData)
    axv.fill_between(model['cyclic_phases'], model['cyclic_rates'], **s.fsModel)
    axv.plot(data['cyclic_phases'], data['cyclic_rates'], **s.lsData)
    axv.plot(model['cyclic_phases'], model['cyclic_rates'], **s.lsModel)
    axc.text(1.9, 0.95, f'$VS^{{(d)}}={data["vectorstrength"]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.text(1.9, 0.83, f'$VS^{{(m)}}={model["vectorstrength"]:.2f}$',
            transform=axc.transAxes, ha='right')
    axv.set_rlim(0, 0.5)
    axv.set_rorigin(-0.2)
    axv.set_xticks_pifracs(4)
    axv.set_yticks_blank()
    # comparison:
    plot_comparison(axb[0], s, '$CV$', np.abs(data['cvbase'] - model['cvbase']))
    axb[0].set_visible(False)
    plot_comparison(axb[1], s, '$CV$', np.abs(data['cvbase'] - model['cvbase']))
    plot_comparison(axb[2], s, '$\\rho_1$', np.abs(data['serialcorrs'][1] - model['serialcorrs'][1]))
    plot_comparison(axb[3], s, '$VS$', np.abs(data['vectorstrength'] - model['vectorstrength']))


def ficurve_data(data_path, cell):
    """ Load fI curve data measured in a cell.

    Parameters
    ----------
    data_path: str
         Base path with the cell data folders.
    cell: str
         Name of the cell its data folder.

    Returns
    -------
    contrasts: ndarray of floats
        Contrasts of step stimuli.
    fonset: ndarray of floats
        Onset spike frequency in Hertz.
    fss: ndarray of floats
        Steady-state spike frequncy in Hertz.
    """
    file_path = os.path.join(data_path, cell, 'fi_curve_info.csv')
    data = pd.read_csv(file_path, index_col=0, comment='#')
    contrasts = data.contrast.to_numpy()
    fonset = data.f_zero.to_numpy()
    fss = data.f_inf.to_numpy()
    return contrasts, fonset, fss


def firate_model(EODf, model_params, contrasts, t0=-0.4, t1=0.2):
    """ Simulate spike frequencies in response to step stimuli.

    Parameters
    ----------
    EODf: float
        EOD frequency in Hertz
    model_params: dict
        Model parameters.
    contrasts: ndarray of floats
        Contrasts for which responses are simulated.
    t0: float
        Start of simulation before the step in seconds (negative).
    t1: float
        End of simulation after the step in seconds.
    """
    deltat = model_params['deltat']
    time = np.arange(t0, t1, deltat)
    rates = [None] * len(contrasts)
    for i, contrast in enumerate(contrasts):
        # generate EOD stimulus with an amplitude step:
        stimulus = np.sin(2*np.pi*EODf*time)
        stimulus[np.argmin(np.abs(time)):] *= (1.0+contrast)
        # integrate the model:
        n = 20
        rate = np.zeros(len(time))
        for k in range(n):
            model_params['v_zero'] = np.random.rand()
            model_params['a_zero'] += 0.02*model_params['a_zero']*np.random.randn()
            spikes = simulate(stimulus, **model_params)
            spikes += time[0]
            trial_rate = instantaneous_rate(spikes, time)
            rate += trial_rate/n
        rates[i] = rate
    return time, rates


def ficurves(time, rates, ton=0.05, tss0=0.1, tss1=0.05, tbase=0.1):
    """ Extract onset and steady-state firing rates from step responses.

    The step stimuli start at time 0 and extend to the end of `time`.

    Parameters
    ----------
    time: ndarray of floats
        Time points of firing rate estimates in `rates`.
        Can be less than the arrays in `rates`.
    rates: list of ndarrays of floats
        Firing rate estimates to be analysed.
    ton: float
        Time interval right after stimulus onset in which to look
        for the onset rate.
    tss0: float
        Start time before stimulus end for estimating steady-state response.
    tss1: float
        End time before stimulus end for estimating steady-state response.
    tbase: float
        Time interval right before stimulus onset for estimating baseline rate.

    Returns
    -------
    fonset: ndarray of floats
        For each response in `rates`, the maximum deviation from
        the baseline rate after stimulus onset.
    fss: ndarray of floats
        For each response in `rates`, the steady-state firing rate
        during the end of the step stimulus, between `tss0` and `tss1`
        before stimulus end.
    baseline: ndarray of floats
        For each response in `rates`, the firing rate during `tbase`
        before the step stimulus.
    """
    baseline = np.zeros(len(rates))
    fonset = np.zeros(len(rates))
    fss = np.zeros(len(rates))
    for i in range(len(rates)):
        rate = rates[i][:len(time)]
        baseline[i] = np.mean(rate[(time > -tbase) & (time < 0)])
        fss[i] = np.mean(rate[(time > time[-1] - tss0) & (time < time[-1] - tss1)])
        fmax = np.max(rate[(time > 0) & (time < ton)])
        fmin = np.min(rate[(time > 0) & (time < ton)])
        fonset[i] = fmax if np.abs(fmax - baseline[i]) > np.abs(fmin - baseline[i]) else fmin
    return fonset, fss, baseline


def sigmoid(x, x0, k, ymax):
    return ymax/(1 + np.exp(-k*(x - x0)))


def fit_ficurves(EODf, contrasts, fonset, fss, data={}):
    fon_contrasts = np.linspace(np.min(contrasts), np.max(contrasts), 200)
    # onset slopes:
    popt = [0.05, 0.03*EODf, EODf]
    try:
        popt, _ = curve_fit(sigmoid, contrasts, fonset, popt)
        fon_slope = popt[2]*popt[1]/4
        fon_line = sigmoid(fon_contrasts, *popt)
    except RuntimeError:
        fon_slope, fon_b = np.polyfit(contrasts, fonset, deg=1)
        fon_line = fon_slope*fon_contrasts + fon_b
    # ss slopes:
    min_contr = -0.2
    max_contr = +0.2
    sel = (contrasts >= min_contr) & (contrasts < max_contr)
    fss_slope, fss_b = np.polyfit(contrasts[sel], fss[sel], deg=1)
    fss_contrasts = np.linspace(min_contr, max_contr, 200)
    fss_line = fss_slope*fss_contrasts + fss_b
    data['fon_slope/Hz'] = fon_slope
    data['fon_contrasts'] = fon_contrasts 
    data['fon_line'] = fon_line 
    data['fss_slope/Hz'] = fss_slope 
    data['fss_contrasts'] = fss_contrasts
    data['fss_line'] = fss_line
    return data

    
def plot_ficurves(ax, axc, s, EODf, data_contrasts, data_fonset, data_fss,
                  model_contrasts, model_fonset, model_fss, data, model):
    """ Plot fI curves. """
    ax.axvline(0, **s.lsGrid)
    ax.axhline(EODf, **s.lsGrid)
    ax.axhline(data['ratebase/Hz'], **s.lsBase)
    ax.plot(100*model_contrasts, model_fss, label=r'$f_{\infty}(I)$', **s.lsSS)
    ax.plot(100*model_contrasts, model_fonset, label=r'$f_0(I)$', **s.lsOnset)
    sel = (data_contrasts >= model_contrasts[0]) & (data_contrasts <= model_contrasts[-1])
    ax.plot(100*data_contrasts[sel], data_fss[sel], **s.psSS)
    ax.plot(100*data_contrasts[sel], data_fonset[sel], **s.psOnset)
    ax.set_xlim(100*model_contrasts[0], 100*model_contrasts[-1])
    ax.set_ylim(0, 1200)
    ax.set_xticks_delta(15)
    ax.set_xlabel('Contrast', '%')
    ax.set_ylabel('Spike frequency', 'Hz')
    # comparison onset slopes:
    ax.plot(100*data['fon_contrasts'], data['fon_line'], **s.lsDataLine)
    ax.plot(100*model['fon_contrasts'], model['fon_line'], **s.lsModelLine)
    plot_comparison(axc[0], s, '$s$', np.abs(model['fon_slope/Hz'] - data['fon_slope/Hz'])/data['fon_slope/Hz'])
    # comparison ss slopes:
    ax.plot(100*data['fss_contrasts'], data['fss_line'], **s.lsDataLine)
    ax.plot(100*model['fss_contrasts'], model['fss_line'], **s.lsModelLine)
    ax.legend(loc='upper left')
    plot_comparison(axc[2], s, '$s$', np.abs(model['fss_slope/Hz'] - data['fss_slope/Hz'])/data['fss_slope/Hz'])
    # comparison chi squared:
    model_idxs = [np.argmin(np.abs(model_contrasts - data_contrasts[i]))
                  for i in range(len(data_contrasts))]
    chifon = np.sum((data_fonset - model_fonset[model_idxs])**2)
    chifss = np.sum((data_fss - model_fss[model_idxs])**2)
    normfon = len(data_contrasts)*(EODf/4)**2
    normfss = len(data_contrasts)*(data['ratebase/Hz']/2)**2
    plot_comparison(axc[1], s, r'$\chi^2 f_{on}$', chifon/normfon)
    plot_comparison(axc[3], s, r'$\chi^2 f_{ss}$', chifss/normfss)

    
def plot_firates(ax, axc, s, contrasts, time, rates):
    """ Plot a few selected firng rates in response to step stimuli. """
    for contrast, rate, style in zip(contrasts, rates, s.rate_styles):
        ax.plot(1000*time, rate, label=f'${100*contrast:+.0f}$%', **style)
    ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.1))
    ax.set_xlim(-20, 50)
    ax.set_ylim(0, 1200)
    ax.set_xlabel('Time', 'ms')
    ax.set_ylabel('Spike frequency', 'Hz')
    #plot_comparison(axc, s, '$\chi^2$', 0.1)


def spectra_model(EODf, model_params, contrasts,
                  fcutoff=300, dt = 0.0005, nfft = 2**9,
                  tinit=0.5, tmax=4.0, trials=20):
    """ Simulate response spectra to whitenoise stimuli.

    Parameters
    ----------
    EODf: float
        EOD frequency in Hertz
    model_params: dict
        Model parameters.
    contrasts: ndarray of floats
        Contrasts for which responses are simulated.
    dt: float
        Temporal resolution for computing spectra in seconds.
    nfft: int
        Number of data points for fourier transform.
    tinit: float
        Initial time of simulation skipped for analysis in seconds.
    tmax: float
        Total time of simulation after tskip in seconds.
    trials: int
        Number of trials.
    """
    deltat = model_params['deltat']
    time = np.arange(0, tinit + tmax, deltat)
    # whitenoise stimulus with stdev = 1:
    rng = np.random.default_rng()
    am = whitenoise(0, fcutoff, dt, tinit + tmax, rng=rng)
    am_interp = np.interp(np.arange(0, tinit + tmax, model_params['deltat']),
                          np.arange(len(am))*dt, am)
    transfers = []
    coheres = []
    spikesc = None
    for i, contrast in enumerate(contrasts):
        # generate EOD stimulus with an amplitude step:
        stimulus = np.sin(2*np.pi*EODf*time)
        stimulus[-len(am_interp):] *= 1 + contrast*am_interp
        spikes = []
        # integrate the model:
        for k in range(trials):
            model_params['v_zero'] = np.random.rand()
            model_params['a_zero'] += 0.02*model_params['a_zero']*np.random.randn()
            spiket = simulate(stimulus, **model_params)
            #spikes.append(spiket[spiket > tinit] - tinit)
            spikes.append(spiket)
        if i == 1:
            spikesc = spikes
        freq, pss, prr, prs = spectra(contrast*am,
                                      spikes, dt, nfft)
        transfers.append(np.abs(prs)/pss)
        coheres.append(np.abs(prs)**2/pss/prr)
    return freq, transfers, coheres, np.arange(len(am))*dt, am, spikes


def analyse_spectra(contrast, freqs, transfer, cohere, fcutoff, model):
    mask = (freqs>0) & (freqs<fcutoff)
    transfer = transfer[mask]
    cohere = cohere[mask]
    freqs = freqs[mask]
    idx = np.argmax(transfer)
    model['contrast'] = contrast
    model['transferfpeak/Hz'] = freqs[idx]
    model['transferpeak/Hz'] = transfer[idx]
    idx = np.argmax(cohere)
    model['coherefpeak/Hz'] = freqs[idx]
    model['coherepeak'] = cohere[idx]
    return model


def plot_transfers(ax, s, freq, transfers, contrasts, fcutoff):
    sel = freq <= fcutoff
    for trans, c, style in zip(transfers, contrasts, s.spectra_styles):
        ax.plot(freq[sel], 1e-2*trans[sel], label=f'{100*c:.0f}%', clip_on=False, **style)
    ax.set_xlabel('Frequency', 'Hz')
    ax.set_ylabel('gain', 'Hz/%')
    ax.set_xlim(0, fcutoff)
    ax.set_xticks_delta(100)
    ax.set_ylim(bottom=0)


def plot_coherences(ax, s, freq, coherences, contrasts, fcutoff):
    sel = freq <= fcutoff
    for cohere, c, style in zip(coherences, contrasts, s.spectra_styles):
        ax.plot(freq[sel], cohere[sel], label=f'{100*c:.0f}%', **style)
    ax.set_xlabel('Frequency', 'Hz')
    ax.set_ylabel('Coherence')
    ax.set_xlim(0, fcutoff)
    ax.set_ylim(0, 1)
    ax.set_xticks_delta(100)
    ax.legend(loc='upper right', markerfirst=True)

    
def plot_raster(ax, s, time, am, frate, fratesd, spikes, twin):
    t0 = 1.0
    sel = (time >= t0) & (time <= t0 + twin)
    spikes = [1e3*(spiket[(spiket > t0) & (spiket < t0 + twin)] - t0) for spiket in spikes]
    rstyle = dict(**s.lsRaster)
    rstyle['linelengths'] *= 1000/len(spikes)
    rstyle['lineoffsets'] *= 1000/len(spikes)
    ax.show_spines('l')
    ax.eventplot(spikes, **rstyle)
    ax.fill_between(1e3*(time[sel] - t0), frate[sel] - fratesd[sel], frate[sel] + fratesd[sel], **s.fsRate)
    ax.plot(1e3*(time[sel] - t0), frate[sel], **s.lsRate)
    ax.set_ylabel('Rate', 'Hz')
    ax.set_ylim(0, 1000)
    ax.xscalebar(1, 1.05, 20, 'ms', ha='right', va='top')
    axs = ax.inset_axes([0, -0.4, 1, 0.35])
    axs.show_spines('')
    axs.axhline(0, **s.lsGrid)
    axs.plot(1e3*(time[sel] - t0), am[sel], clip_on=False, **s.lsStim)
    axs.text(-5, 0, r'10\,\%', ha='right', va='center')
    
    
def check_baseeod(data_path, cell):
    """
    bad EODs (clipped EOD trace!)
    2012-06-27-ah-invivo-1 

    bad EODs (big chirps!!!):
    2014-06-06-ac-invivo-1         CV=0.188
    2014-06-06-ag-invivo-1         CV=0.195
    2018-06-25-ad-invivo-1         many small chirps
    2018-06-26-ah-invivo-1         CV=0.203   many big chirps
    """
    data_eods = np.load(os.path.join(data_path, cell, 'baseline_eods_trial_1.npy'))
    fig, ax = plt.subplots()
    ax.set_title(cell)
    rate = 1/np.diff(data_eods)
    ax.plot(data_eods[:-1], rate)
    #plot_eod_interval_hist(ax, data_eods, max_iei=5)
    plt.show()


def main(model_path):
    if model_path is None:
        model_path = '../models.csv'
    
    data_path = '../celldata'
    plot_path = 'plots'
    suffix = ''
    if '_' in model_path:
        suffix = '_' + model_path.split('_')[-1].split('.')[0]
    plot_path += suffix
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
        
    s = plot_style()

    baseline_tmax = 15 # seconds
    model_contrasts = np.arange(-0.3, 0.31, 0.01)
    rate_contrasts = [-0.2, +0.05, -0.1, +0.1, -0.05, +0.2]
    s.rate_styles = [s.lsNegContrast,
                     pt.lighter(s.lsPosContrast, 0.4),
                     pt.lighter(s.lsNegContrast, 0.7),
                     pt.lighter(s.lsPosContrast, 0.7),
                     pt.lighter(s.lsNegContrast, 0.4),
                     s.lsPosContrast]
    spectra_contrasts = [0.05, 0.1, 0.2]
    s.spectra_styles = [pt.lighter(s.lsSpec, 0.4), pt.lighter(s.lsSpec, 0.7), s.lsSpec]
    fcutoff = 300
    data_dicts = []
    model_dicts = []
    
    # load model parameter:
    parameters = load_models(model_path)

    # loop over model cells:
    for cell_idx in range(len(parameters)):
    #for cell_idx in [-1]:
        model_params = parameters[cell_idx]
        cell = model_params.pop('cell')
        name = model_params.pop('name', '')
        EODf = model_params.pop('EODf')
        print(f'cell {cell_idx:3d}: {cell} {name}')
        data = dict(cell=cell)
        model = dict(cell=cell)
        #check_baseeod(data_path, cell)
        #continue

        # setup figure:
        fig, axg = plt.subplots(2, 2, cmsize=(16, 15), width_ratios=[42, 2],
                                height_ratios=[3, 1])
        fig.subplots_adjust(leftm=8.5, rightm=3, bottomm=4.5, topm=4,
                            wspace=0.07, hspace=0.25)
        fig.text(0.03, 0.97, f'{cell} {name}', ha='left', fontsize='large')
        fig.text(0.5, 0.97, f'{os.path.basename(model_path)} {cell_idx}', ha='center', color=s.colors['gray'])
        fig.text(0.97, 0.97, f'EOD$f$={EODf:.0f}Hz', ha='right', fontsize='large')
        axs = axg[0, 0].subplots(2, 3, wspace=0.8, hspace=0.4)
        axs[0, 2] = axs[0, 2].make_polar(-0.02, -0.05)
        axr = fig.merge(axs[1, 1:3])
        axc = axg[0, 1].subplots(4, 2, hspace=0.6, wspace=0.2)
        axg[1, 1].set_visible(False)
        axn = axg[1, 0].subplots(1, 3, width_ratios=[3, 2, 2], wspace=0.6)

        # baseline:
        data_spikes, data_eods = baseline_data(data_path, cell)
        model_spikes = baseline_model(EODf, model_params, baseline_tmax)
        data = analyse_baseline(EODf, data_spikes, data_eods, data,
                                max_eods=15.5, max_lag=15)
        model = analyse_baseline(EODf, model_spikes, baseline_tmax, model,
                                 max_eods=15.5, max_lag=15)
        plot_baseline(axs[0, 0], axs[0, 1], axs[0, 2], axc[0:2,:].ravel(), s,
                      EODf, data, model)
        
        # fi curves:
        data_contrasts, data_fonset, data_fss = ficurve_data(data_path, cell)
        time, rates = firate_model(EODf, model_params, model_contrasts)
        model_fonset, model_fss, baseline = ficurves(time, rates)
        data = fit_ficurves(EODf, data_contrasts, data_fonset, data_fss, data)
        model = fit_ficurves(EODf, model_contrasts, model_fonset, model_fss,
                             model)
        plot_ficurves(axs[1, 0], axc[2:4,:].ravel(), s, EODf,
                      data_contrasts, data_fonset, data_fss,
                      model_contrasts, model_fonset, model_fss, data, model)
        
        # fi rates:
        time, rates = firate_model(EODf, model_params, rate_contrasts)
        plot_firates(axr, None, s, rate_contrasts, time, rates)

        # spectra:
        freq, transfers, coheres, time, am, spikes = \
            spectra_model(EODf, model_params, spectra_contrasts,
                          fcutoff, 0.0005, 2**9, tinit=0.5, tmax=20.0,
                          trials=20)
        for ksd in [1, 2, 4]:
            frate, fratesd = rate(time, spikes, 0.001*ksd)
            model[f'respmod{ksd}/Hz'] = np.std(frate)
        model = analyse_spectra(spectra_contrasts[1], freq, transfers[1],
                                coheres[1], fcutoff, model)
        frate, fratesd = rate(time, spikes, 0.001)
        plot_raster(axn[0], s, time, am, frate, fratesd, spikes, 0.1)
        plot_transfers(axn[1], s, freq, transfers, spectra_contrasts, fcutoff)
        plot_coherences(axn[2], s, freq, coheres, spectra_contrasts, fcutoff)

        fig.common_yspines(axc)
        file_name = cell
        if name:
            file_name = file_name + '-' + name
        fig.savefig(os.path.join(plot_path, file_name))
        plt.close(fig)
        #plt.show()

        data_dicts.append(data)
        model_dicts.append(model)

        #if len(model_dicts) > 3:
        #    break

    data = pd.DataFrame(data_dicts)
    model = pd.DataFrame(model_dicts)
    data = data.drop(columns=['isis', 'hist', 'lags', 'cyclic_phases', 'cyclic_rates', 'fon_contrasts', 'fon_line', 'fss_contrasts', 'fss_line'])
    data.serialcorrs = [c[1] for c in data.serialcorrs]
    data.rename(columns=dict(serialcorrs='serialcorr1'), inplace=True)
    model = model.drop(columns=['isis', 'hist', 'lags', 'cyclic_phases', 'cyclic_rates', 'fon_contrasts', 'fon_line', 'fss_contrasts', 'fss_line'])
    model.serialcorrs = [c[1] for c in model.serialcorrs]
    model.rename(columns=dict(serialcorrs='serialcorr1'), inplace=True)
    data.to_csv('punit' + suffix + 'data.csv', index_label='index')
    model.to_csv('model' + suffix + 'data.csv', index_label='index')

        
if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else None)
