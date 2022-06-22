import numpy as np
try:
    from numba import jit
except ImportError:
    def jit(nopython):
        def decorator_jit(func):
            return func
        return decorator_jit


def load_models(file):
    """ Load model parameter from csv file.

    Parameters
    ----------
    file: string
        Name of file with model parameters.

    Returns
    -------
    parameters: list of dict
        For each cell a dictionary with model parameters.
    """
    parameters = []
    with open(file, 'r') as file:
        header_line = file.readline()
        header_parts = header_line.strip().split(",")
        keys = header_parts
        for line in file:
            line_parts = line.strip().split(",")
            parameter = {}
            for i in range(len(keys)):
                parameter[keys[i]] = float(line_parts[i]) if i > 0 else line_parts[i]
            parameters.append(parameter)
    return parameters


@jit(nopython=True)
def simulate(stimulus, deltat=0.00005, v_zero=0.0, a_zero=2.0, threshold=1.0, v_base=0.0,
             delta_a=0.08, tau_a=0.1, v_offset=-10.0, mem_tau=0.015, noise_strength=0.05,
             input_scaling=60.0, dend_tau=0.001, ref_period=0.001):
    """ Simulate a P-unit.

    Returns
    -------
    spike_times: 1-D array
        Simulated spike times in seconds.
    """    
    # initial conditions:
    v_dend = stimulus[0]
    v_mem = v_zero
    adapt = a_zero

    # prepare noise:    
    noise_strength_sqrt = np.sqrt(noise_strength * 2)
    noise = np.random.randn(len(stimulus))
    noise *= noise_strength_sqrt / np.sqrt(deltat)

    # rectify stimulus array:
    stimulus = stimulus.copy()
    stimulus[stimulus < 0.0] = 0.0

    # integrate:
    spike_times = []
    for i in range(len(stimulus)):
        v_dend += (-v_dend + stimulus[i]) / dend_tau * deltat
        v_mem += (v_base - v_mem + v_offset + (
                    v_dend * input_scaling) - adapt + noise[i]) / mem_tau * deltat
        adapt += -adapt / tau_a * deltat

        # refractory period:
        if len(spike_times) > 0 and (deltat * i) - spike_times[-1] < ref_period + deltat/2:
            v_mem = v_base

        # threshold crossing:
        if v_mem > threshold:
            v_mem = v_base
            spike_times.append(i * deltat)
            adapt += delta_a / tau_a

    return np.array(spike_times)
