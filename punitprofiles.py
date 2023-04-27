import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from model import simulate, load_models
import plottools.plottools as pt


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


def baseline_vectorstrength(spikes, period, sigma=0.05, deltat=0.0001):
    time = np.arange(0, period, 0.01*period)
    phases = 2*np.pi*time/period
    rate = np.zeros(len(time))
    n = 0
    vectors = np.zeros(len(spikes), dtype=complex)
    for i, spike in enumerate(spikes):
        cycle = (spike // period)*period
        cycle_time = spike - cycle
        phase = 2*np.pi*cycle_time/period
        vectors[i] = np.exp(1j*phase)
        cycle_spikes = spikes[(spikes - cycle >= -10*period) & (spikes - cycle <= 10*period)] - cycle
        if len(cycle_spikes) > 1:
            kernel = gaussian_kde(cycle_spikes, sigma*period/np.std(cycle_spikes, ddof=1))
            cycle_rate = kernel(time)
            rate += cycle_rate
            n += 1
    vs = np.abs(np.mean(vectors))
    return phases, rate/n, vs
        

def baseline_serialcorr(spikes, max_lag=10):
    isis = np.diff(spikes)
    lags = np.arange(0, max_lag, 1)
    corrs = np.zeros(max_lag)
    corrs[0] = np.corrcoef(isis, isis)[0,1]
    for i, lag in enumerate(lags[1:]):
        corrs[i+1] = np.corrcoef(isis[:-lag], isis[lag:])[0,1]
    return lags, corrs

    
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


def base_profile(axi, axc, axv, cell, EODf, model_params):
    data_color = 'tab:blue'
    model_color = 'tab:red'
    lsGrid = dict(color='gray', ls=':')
    lsData = dict(color=data_color, lw=2)
    lsModel = dict(color=model_color, lw=2)
    lpsData = dict(ls='-', color=data_color, lw=2, marker='o', ms=8, clip_on=False)
    lpsModel = dict(ls='-', color=model_color, lw=2, marker='o', ms=8, clip_on=False)
    fsData = dict(color=data_color, alpha=0.5)
    fsModel = dict(color=model_color, alpha=0.5)
    max_eods = 15.5
    max_lag = 16
    eod_period = 1/EODf
    # data:
    data_spikes = np.load(f'celldata/{cell}/baseline_spikes_trial_1.npy')
    data_isi, data_kde, data_rate, data_cv = \
        baseline_isih(data_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    data_cphase, data_crate, data_vs = baseline_vectorstrength(data_spikes, eod_period, sigma=0.02)
    data_lags, data_corrs = baseline_serialcorr(data_spikes, max_lag=max_lag)
    # model:
    deltat = model_params["deltat"]
    time = np.arange(0, 30, deltat)
    # baseline EOD with amplitude 1:
    stimulus = np.sin(2*np.pi*EODf*time)
    # integrate the model:
    model_spikes = simulate(stimulus, **model_params)
    model_isi, model_kde, model_rate, model_cv = \
        baseline_isih(model_spikes, sigma=0.05*eod_period, maxisi=max_eods*eod_period)
    model_cphase, model_crate, model_vs = baseline_vectorstrength(model_spikes, eod_period, sigma=0.02)
    model_lags, model_corrs = baseline_serialcorr(model_spikes, max_lag=max_lag)
    # plot isih statistics:
    axi.show_spines('lb')
    for eod in np.arange(1, max_eods, 1):
        axi.axvline(1000*eod*eod_period, **lsGrid)
    axi.fill_between(1000*data_isi, data_kde/1000, **fsData)
    axi.fill_between(1000*model_isi, model_kde/1000, **fsModel)
    axi.plot(1000*data_isi, data_kde/1000, **lsData)
    axi.plot(1000*model_isi, model_kde/1000, **lsModel)
    axi.text(0.75, 0.95, f'$r={data_rate:.0f}$Hz',
            transform=axi.transAxes, ha='right')
    axi.text(0.95, 0.95, f'$CV_d={data_cv:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.text(0.95, 0.85, f'$CV_m={model_cv:.2f}$',
            transform=axi.transAxes, ha='right')
    axi.set_xlabel('ISI [ms]')
    axi.set_ylabel('pdf [1/ms]')
    # plot serial correlations:
    axc.show_spines('lb')
    axc.axhline(0, **lsGrid)
    axc.plot(data_lags, data_corrs, **lpsData)
    axc.plot(model_lags, model_corrs, **lpsModel)
    axc.text(0.95, 0.95, f'$\\rho_d={data_corrs[1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.text(0.95, 0.85, f'$\\rho_m={model_corrs[1]:.2f}$',
            transform=axc.transAxes, ha='right')
    axc.set_ylim(-1, 1)
    axc.set_xlabel('lag')
    axc.set_ylabel('correlation')
    # plot vector strength:
    axv.fill_between(data_cphase, data_crate, **fsData)
    axv.fill_between(model_cphase, model_crate, **fsModel)
    axv.plot(data_cphase, data_crate, **lsData)
    axv.plot(model_cphase, model_crate, **lsModel)
    axv.text(1, 1, f'$VS_d={data_vs:.2f}$',
            transform=axv.transAxes, ha='right')
    axv.text(1, 0.9, f'$VS_m={model_vs:.2f}$',
            transform=axv.transAxes, ha='right')
    axv.set_rlim(0, 1200)
    axv.set_rorigin(-400)
    axv.set_xticks_pifracs(4)
    #axv.set_theta
    


def main():
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0
    
    # load model parameter:
    parameters = load_models("models.csv")

    for example_cell_idx in range(len(parameters)):

        model_params = parameters[example_cell_idx]
        cell = model_params.pop('cell')
        EODf = model_params.pop('EODf')
        print("cell:", cell)
        fig, axs = plt.subplots(1, 3)
        axs[2] = axs[2].make_polar()
        base_profile(axs[0], axs[1], axs[2], cell, EODf, model_params)
        plt.show()
        #break
    
        """

        # generate EOD-like stimulus with an amplitude step:
        deltat = model_params["deltat"]
        stimulus_length = 2.0  # in seconds
        time = np.arange(0, stimulus_length, deltat)
        # baseline EOD with amplitude 1:
        stimulus = np.sin(2*np.pi*EODf*time)
        # amplitude step with given contrast:
        t0 = 0.5
        t1 = 1.5
        contrast = 0.3
        stimulus[int(t0//deltat):int(t1//deltat)] *= (1.0+contrast)

        # integrate the model:
        spikes = simulate(stimulus, **model_params)

        # some analysis and plotting:
        rate = instantaneous_rate(spikes, time)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex="col")

        ax1.plot(time, stimulus)
        ax1.set_title("Stimulus")
        ax1.set_ylabel("Amplitude in mV")

        ax2.plot(time, rate)
        ax2.set_title("Model Frequency")
        ax2.set_ylabel("Frequency in Hz")
        ax2.set_xlabel("Time in s")
        plt.show()
        """


if __name__ == '__main__':
    main()
