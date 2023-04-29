import os
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
from model import simulate, load_models
from eods import plot_eod_interval_hist
import plottools.plottools as pt


def plot_style():
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


def baseline_isih(spikes, sigma=1e4, maxisi=0.1):
    isis = np.diff(spikes)
    isi = np.arange(0.0, maxisi, 0.1*sigma)
    kernel = gaussian_kde(isis, sigma/np.std(isis, ddof=1))
    kde = kernel(isi)
    mean_isi = np.mean(isis)
    std_isi = np.std(isis)
    rate = 1/mean_isi
    cv = std_isi / mean_isi
    return isi, kde, rate, cv


def baseline_vectorstrength(spikes, eods, sigma=0.05):
    phases = np.arange(0, 1.005, 0.005)*2*np.pi
    rate = np.zeros(len(phases))
    n = 0
    vectors = np.zeros(len(spikes), dtype=complex)
    for i, spike in enumerate(spikes):
        k = eods.searchsorted(spike) - 1
        if k + 1 >= len(eods):
            continue
        cycle = eods[k]
        period = eods[k+1] - eods[k]
        phase = 2*np.pi*(spike - cycle)/period
        vectors[i] = np.exp(1j*phase)
        cycle_spikes = np.array([phase - 2*np.pi, phase, phase + 2*np.pi])
        kernel = gaussian_kde(cycle_spikes, 2*np.pi*sigma/np.std(cycle_spikes, ddof=1))
        cycle_rate = kernel(phases)
        """
        if i == 10:
            print(np.max(cycle_rate))
            plt.close('all')
            plt.plot(phases, cycle_rate)
            plt.show()
        """
        rate += cycle_rate
        n += 1
    vs = np.abs(np.mean(vectors))
    """
    print(n, np.max(rate/n))
    plt.close('all')
    plt.plot(phases, rate/n)
    plt.show()
    """
    return phases, rate/n, vs
        

def baseline_serialcorr(spikes, max_lag=10):
    isis = np.diff(spikes)
    lags = np.arange(0, max_lag, 1)
    corrs = np.zeros(max_lag)
    corrs[0] = np.corrcoef(isis, isis)[0,1]
    for i, lag in enumerate(lags[1:]):
        corrs[i+1] = np.corrcoef(isis[:-lag], isis[lag:])[0,1]
    rng = np.random.default_rng()
    perm_corrs = np.zeros(10000)
    for k in range(len(perm_corrs)):
        xisis = rng.permutation(isis)
        yisis = rng.permutation(isis)
        perm_corrs[k] = np.corrcoef(xisis, yisis)[0,1]
    low, high = np.quantile(perm_corrs, (0.001, 0.999))
    return lags, corrs, low, high


def load_ficurve(file_path):
    data = pd.read_csv(file_path, index_col=0)
    contrast = data.contrast.to_numpy()
    fonset = data.f_zero.to_numpy()
    fss = data.f_inf.to_numpy()
    return contrast, fonset, fss

    
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


def base_profile(axi, axc, axv, s, cell, EODf, model_params):
    max_eods = 15.5
    max_lag = 16
    eod_period = 1/EODf
    # data:
    data_spikes = np.load(f'celldata/{cell}/baseline_spikes_trial_1.npy')
    data_eods = np.load(f'celldata/{cell}/baseline_eods_trial_1.npy')
    data_isi, data_kde, data_rate, data_cv = \
        baseline_isih(data_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    data_cphase, data_crate, data_vs = baseline_vectorstrength(data_spikes, data_eods, sigma=0.02)
    data_lags, data_corrs, data_low, data_high = baseline_serialcorr(data_spikes, max_lag=max_lag)
    # model:
    deltat = model_params["deltat"]
    time = np.arange(0, 15, deltat)
    # baseline EOD with amplitude 1:
    stimulus = np.sin(2*np.pi*EODf*time)
    # integrate the model:
    model_spikes = simulate(stimulus, **model_params)
    model_isi, model_kde, model_rate, model_cv = \
        baseline_isih(model_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    model_cphase, model_crate, model_vs = baseline_vectorstrength(model_spikes, np.arange(0, time[-1] + 2*eod_period, eod_period), sigma=0.02)
    model_lags, model_corrs, model_low, model_high = baseline_serialcorr(model_spikes, max_lag=max_lag)
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
    #axv.set_theta
    return data_rate

    
def ficurves(axf, axr, s, cell, EODf, model_params, base_rate):
    # data:
    data_contrast, data_fonset, data_fss = load_ficurve(f'celldata/{cell}/fi_curve_info.csv')
    # model:
    plot_contrasts = [-0.2, +0.05, -0.1, +0.1, -0.05, +0.2]
    plot_rates = [None] * len(plot_contrasts)
    model_contrast = np.arange(-0.3, 0.31, 0.01)
    model_fonset = np.zeros(len(model_contrast))
    model_fss = np.zeros(len(model_contrast))
    for i, contrast in enumerate(model_contrast):
        # generate EOD stimulus with an amplitude step:
        deltat = model_params["deltat"]
        time = np.arange(-0.4, 0.2, deltat)
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
        fss = np.mean(rate[(time > 0.1) & (time < 0.15)])
        fmax = np.max(rate[(time > 0) & (time < 0.05)])
        fmin = np.min(rate[(time > 0) & (time < 0.05)])
        fon = fmax if np.abs(fmax - fss) > np.abs(fmin - fss) else fmin
        model_fonset[i] = fon
        model_fss[i] = fss
        for k in range(len(plot_contrasts)):
            if np.abs(contrast - plot_contrasts[k]) < 1e-3:
                plot_rates[k] = rate
        
    # plot ficurves:
    axf.axvline(0, **s.lsGrid)
    axf.axhline(EODf, **s.lsGrid)
    axf.axhline(base_rate, **s.lsBase)
    axf.plot(100*model_contrast, model_fss, **s.lsSS)
    axf.plot(100*model_contrast, model_fonset, **s.lsOnset)
    axf.plot(100*data_contrast, data_fss, **s.psSS)
    axf.plot(100*data_contrast, data_fonset, **s.psOnset)
    axf.set_xlim(-30, 30)
    axf.set_ylim(0, 1200)
    axf.set_xticks_delta(15)
    axf.set_xlabel('Contrast', '%')
    axf.set_ylabel('Spike frequency', 'Hz')
    # plot fi rates:
    plot_colors = {+0.20: s.lsPosContrast,
                   +0.10: pt.lighter(s.lsPosContrast, 0.7),
                   +0.05: pt.lighter(s.lsPosContrast, 0.4),
                   -0.05: pt.lighter(s.lsNegContrast, 0.4),
                   -0.10: pt.lighter(s.lsNegContrast, 0.7),
                   -0.20: s.lsNegContrast}
    for contrast, rate in zip(plot_contrasts, plot_rates):
        axr.plot(1000*time, rate, label=f'${100*contrast:+.0f}$%',
                 **plot_colors[contrast])
    axr.legend(ncol=3, loc='upper right', bbox_to_anchor=(1, 1.1))
    axr.set_xlim(-20, 50)
    axr.set_ylim(0, 1200)
    axr.set_xlabel('Time', 'ms')
    axr.set_ylabel('Spike frequency', 'Hz')


def check_baseeod(cell):
    """
    bad EODs:
2014-06-06-ac-invivo-1         CV=0.188
2014-06-06-ag-invivo-1         CV=0.195
2018-06-26-ah-invivo-1         CV=0.203
    """
    data_eods = np.load(f'celldata/{cell}/baseline_eods_trial_1.npy')
    fig, ax = plt.subplots()
    ax.set_title(cell)
    plot_eod_interval_hist(ax, data_eods, max_iei=5)
    plt.show()


def main():
    s = plot_style()

    plot_path = 'plots'
    if not os.path.isdir(plot_path):
        os.mkdir(plot_path)
    
    # load model parameter:
    parameters = load_models("models.csv")

    for example_cell_idx in range(len(parameters)):

        model_params = parameters[example_cell_idx]
        cell = model_params.pop('cell')
        EODf = model_params.pop('EODf')
        print("cell:", cell)
        check_baseeod(cell)
        continue
        fig, axs = plt.subplots(2, 3, cmsize=(16, 11))
        fig.subplots_adjust(leftm=8, rightm=2, bottomm=3.5, topm=3,
                            wspace=0.8, hspace=0.4)
        fig.text(0.03, 0.96, cell, ha='left')
        fig.text(0.97, 0.96, f'EOD$f$={EODf:.0f}Hz', ha='right')
        axs[0, 2] = axs[0, 2].make_polar(-0.02, -0.05)
        axr = fig.merge(axs[1,1:3])
        data_rate = 200
        data_rate = base_profile(axs[0, 0], axs[0, 1], axs[0, 2], s,
                                 cell, EODf, model_params)
        ficurves(axs[1, 0], axr, s, cell, EODf, model_params, data_rate)
        fig.savefig(os.path.join(plot_path, cell))
        plt.close(fig)
        #plt.show()
        #break

        
if __name__ == '__main__':
    main()
