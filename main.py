import numpy as np
import matplotlib.pyplot as plt

from model import simulate, load_models

"""
Dependencies:
numpy 
matplotlib
numba (optional, speeds simulation up: pre-compiles functions to machine code)
"""


def main():
    # tiny example program:

    example_cell_idx = 20

    # load model parameter:
    parameters = load_models("models.csv")
    #parameters = load_models("models_202106.csv")

    for example_cell_idx in range(len(parameters)):

        model_params = parameters[example_cell_idx]
        cell = model_params.pop('cell')
        EODf = model_params.pop('EODf')
        print("Example with cell:", cell)

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
    plt.close()


    
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


if __name__ == '__main__':
    main()
