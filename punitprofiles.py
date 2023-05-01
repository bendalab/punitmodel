import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model import simulate, load_models
from eods import plot_eod_interval_hist
from baseline import interval_statistics, serial_correlations, vector_strength, cyclic_rate
import plottools.plottools as pt


def plot_style():
    """ Settings and styles for the plots.

    Returns
    -------
    s: namespace
        Plotting styles (dictionaries with colors, line poperties, etc.).
    """
    class s:
        colors = pt.palettes['muted']
        data_color = colors['blue']
        model_color = colors['red']
        onset_color = colors['green']
        ss_color = colors['red']
        base_color = colors['lightblue']
        poscontrast_color = colors['magenta']
        negcontrast_color = colors['purple']
        
        lw = 1.5
        ms = 4
        
        lsGrid = dict(color='gray', ls=':', lw=0.5)
        lsData = dict(color=data_color, lw=lw)
        lsModel = dict(color=model_color, lw=lw)
        lpsData = dict(ls='-', color=data_color, lw=lw, marker='o', ms=ms, clip_on=False)
        lpsModel = dict(ls='-', color=model_color, lw=lw, marker='o', ms=ms, clip_on=False)
        fsData = dict(color=data_color, alpha=0.5)
        fsModel = dict(color=model_color, alpha=0.5)
        psOnset = dict(ls='none', color=onset_color, marker='o', ms=ms, clip_on=False)
        psSS = dict(ls='none', color=ss_color, marker='o', ms=ms, clip_on=False)
        lsOnset = dict(ls='-', color=onset_color, lw=lw)
        lsSS = dict(ls='-', color=ss_color, lw=lw)
        lsBase = dict(ls='-', color=base_color, lw=lw)

        lsPosContrast = dict(ls='-', color=poscontrast_color, lw=lw)
        lsNegContrast = dict(ls='-', color=negcontrast_color, lw=lw)
    pt.axes_params(0, 0, 0, 'none')
    pt.figure_params(format='pdf')
    pt.spines_params('lb')
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
    deltat = model_params["deltat"]
    time = np.arange(0, tmax, deltat)
    # baseline EOD with amplitude 1:
    stimulus = np.sin(2*np.pi*EODf*time)
    # integrate the model:
    spikes = simulate(stimulus, **model_params)
    return spikes


def plot_baseline(axi, axc, axv, s, EODf,
                  data_spikes, data_eods, model_spikes):
    """ Compute and plot baseline statistics for data and model spikes. """
    max_eods = 15.5
    max_lag = 15
    eod_period = 1/EODf
    # analyse data:
    data_isi, data_kde, data_rate, data_cv = \
        interval_statistics(data_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    data_lags, data_corrs, data_low, data_high = serial_correlations(data_spikes, max_lag=max_lag)
    data_vs = vector_strength(data_spikes, data_eods)
    data_cphase, data_crate = cyclic_rate(data_spikes, data_eods, sigma=0.02)
    # analyse model:
    model_isi, model_kde, model_rate, model_cv = \
        interval_statistics(model_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    model_lags, model_corrs, model_low, model_high = serial_correlations(model_spikes, max_lag=max_lag)
    model_vs = vector_strength(model_spikes, eod_period)
    model_cphase, model_crate = cyclic_rate(model_spikes, eod_period, sigma=0.02)
    # plot isih statistics:
    for eod in np.arange(1, max_eods, 1):
        axi.axvline(1000*eod*eod_period, **s.lsGrid)
    axi.fill_between(1000*data_isi, data_kde/1000, **s.fsData)
    axi.fill_between(1000*model_isi, model_kde/1000, **s.fsModel)
    axi.plot(1000*data_isi, data_kde/1000, **s.lsData)
    axi.plot(1000*model_isi, model_kde/1000, **s.lsModel)
    axi.text(1, 0.95, f'$r={data_rate:.0f}$Hz',
            transform=axi.transAxes, ha='right')
    axi.text(1, 0.83, f'$CV_d={data_cv:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.text(1, 0.71, f'$CV_m={model_cv:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.set_xlabel('ISI', 'ms')
    axi.set_ylabel('pdf', '1/ms')
    axi.set_xticks_delta(10)
    # plot serial correlations:
    axc.axhspan(data_low, data_high, **s.fsData)
    axc.axhline(0, **s.lsGrid)
    axc.plot(data_lags, data_corrs, label='data', **s.lpsData)
    axc.plot(model_lags, model_corrs, label='model', **s.lpsModel)
    axc.text(1, 0.95, f'$\\rho_{{1,d}}={data_corrs[1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.text(1, 0.83, f'$\\rho_{{1,m}}={model_corrs[1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.set_ylim(-1, 1)
    axc.set_xlabel('Lag $k$')
    axc.set_ylabel('Correlation $\\rho_k$')
    axc.set_xticks_delta(5)
    axc.legend(loc='lower right', markerfirst=False)
    # plot vector strength:
    axv.fill_between(data_cphase, data_crate, **s.fsData)
    axv.fill_between(model_cphase, model_crate, **s.fsModel)
    axv.plot(data_cphase, data_crate, **s.lsData)
    axv.plot(model_cphase, model_crate, **s.lsModel)
    axc.text(1.9, 0.95, f'$VS_d={data_vs:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.text(1.9, 0.83, f'$VS_m={model_vs:.2f}$',
            transform=axc.transAxes, ha='right')
    axv.set_rlim(0, 0.5)
    axv.set_rorigin(-0.1)
    axv.set_xticks_pifracs(4)
    axv.set_yticks_blank()
    return data_rate


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
    data = pd.read_csv(file_path, index_col=0)
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
    deltat = model_params["deltat"]
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

    
def plot_ficurves(ax, s, EODf, data_contrasts, data_fonset, data_fss,
                  model_contrasts, model_fonset, model_fss, base_rate):
    """ Plot fI curves. """
    ax.axvline(0, **s.lsGrid)
    ax.axhline(EODf, **s.lsGrid)
    ax.axhline(base_rate, **s.lsBase)
    ax.plot(100*model_contrasts, model_fss, **s.lsSS)
    ax.plot(100*model_contrasts, model_fonset, **s.lsOnset)
    sel = (data_contrasts >= model_contrasts[0]) & (data_contrasts <= model_contrasts[-1])
    ax.plot(100*data_contrasts[sel], data_fss[sel], **s.psSS)
    ax.plot(100*data_contrasts[sel], data_fonset[sel], **s.psOnset)
    ax.set_xlim(100*model_contrasts[0], 100*model_contrasts[-1])
    ax.set_ylim(0, 1200)
    ax.set_xticks_delta(15)
    ax.set_xlabel('Contrast', '%')
    ax.set_ylabel('Spike frequency', 'Hz')

    
def plot_firates(ax, s, contrasts, time, rates):
    """ Plot a few selected firng rates in response to step stimuli. """
    for contrast, rate, color in zip(contrasts, rates, s.rate_colors):
        ax.plot(1000*time, rate, label=f'${100*contrast:+.0f}$%', **color)
    ax.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.1))
    ax.set_xlim(-20, 50)
    ax.set_ylim(0, 1200)
    ax.set_xlabel('Time', 'ms')
    ax.set_ylabel('Spike frequency', 'Hz')


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


def main():
    s = plot_style()

    data_path = 'celldata'
    plot_path = 'plots'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)

    baseline_tmax = 15 # seconds
    model_contrasts = np.arange(-0.3, 0.31, 0.01)
    rate_contrasts = [-0.2, +0.05, -0.1, +0.1, -0.05, +0.2]
    s.rate_colors = [s.lsNegContrast,
                     pt.lighter(s.lsPosContrast, 0.4),
                     pt.lighter(s.lsNegContrast, 0.7),
                     pt.lighter(s.lsPosContrast, 0.7),
                     pt.lighter(s.lsNegContrast, 0.4),
                     s.lsPosContrast]
    
    # load model parameter:
    parameters = load_models("models.csv")

    for example_cell_idx in range(len(parameters)):

        model_params = parameters[example_cell_idx]
        cell = model_params.pop('cell')
        EODf = model_params.pop('EODf')
        print("cell:", cell)
        #check_baseeod(data_path, cell)
        #continue
        fig, axs = plt.subplots(2, 3, cmsize=(16, 11))
        fig.subplots_adjust(leftm=8, rightm=2, bottomm=3.5, topm=3,
                            wspace=0.8, hspace=0.4)
        fig.text(0.03, 0.96, cell, ha='left')
        fig.text(0.97, 0.96, f'EOD$f$={EODf:.0f}Hz', ha='right')
        axs[0, 2] = axs[0, 2].make_polar(-0.02, -0.05)
        axr = fig.merge(axs[1,1:3])
        # baseline:
        data_spikes, data_eods = baseline_data(data_path, cell)
        model_spikes = baseline_model(EODf, model_params, baseline_tmax)
        data_rate = 200
        data_rate = plot_baseline(axs[0, 0], axs[0, 1], axs[0, 2], s, EODf,
                                  data_spikes, data_eods, model_spikes)
        # fi curves:
        data_contrasts, data_fonset, data_fss = ficurve_data(data_path, cell)
        time, rates = firate_model(EODf, model_params, model_contrasts)
        model_fonset, model_fss, baseline = ficurves(time, rates)
        plot_ficurves(axs[1, 0], s, EODf, data_contrasts, data_fonset, data_fss,
                      model_contrasts, model_fonset, model_fss, data_rate)
        # fi rates:
        time, rates = firate_model(EODf, model_params, rate_contrasts)
        plot_firates(axr, s, rate_contrasts, time, rates)
        #fig.savefig(os.path.join(plot_path, cell))
        #plt.close(fig)
        plt.show()
        #break

        
if __name__ == '__main__':
    main()
