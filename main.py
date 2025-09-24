
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

    # some analysis an dplotting:
    freq = calculate_isi_frequency(spikes, deltat)
    freq_time = np.arange(spikes[0], spikes[-1], deltat)

    fig, axs = plt.subplots(2, 1, sharex="col")

    axs[0].plot(time, stimulus)
    axs[0].set_title("Stimulus")
    axs[0].set_ylabel("Amplitude in mV")

    axs[1].plot(freq_time, freq)
    axs[1].set_title("Model Frequency")
    axs[1].set_ylabel("Frequency in Hz")
    axs[1].set_xlabel("Time in s")
    plt.show()
    plt.close()


def calculate_isi_frequency(spikes, deltat):
    """
    calculates inter-spike interval frequency
    (wasn't tested a lot may give different length than time = np.arange(spikes[0], spikes[-1], deltat),
    or raise an index error for some inputs)

    :param spikes: spike time points
    :param deltat: integration time step of the model

    :return: the frequency trace:
                starts at the time of first spike
                ends at the time of the last spike.
    """

    isis = np.diff(spikes)
    freq_points = 1 / isis
    freq = np.zeros(int((spikes[-1] - spikes[0]) / deltat))

    current_idx = 0
    for i, isi in enumerate(isis):
        end_idx = int(current_idx + np.rint(isi / deltat))
        freq[current_idx:end_idx] = freq_points[i]
        current_idx = end_idx

    return freq


if __name__ == '__main__':
    main()
