import os
import sys
import glob
import gzip
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.dataloader import relacs_samplerate_unit, relacs_metadata
sys.path.insert(0, os.path.join(sys.path[0], '..'))
from eods import detect_eods, plot_eod_interval_hist

# select what to do:
plot_data_trace = False
plot_data_hist = False
plot_detection = False
check_iei_hist = False
check_cv = True
save_eods = True


data_path = 'data/'

for cell_path in sorted(glob.glob('2*')):
    if not os.path.isdir(os.path.join(data_path, cell_path)):
        continue
    if not check_cv:
        print(cell_path)
    samplerate, _ = relacs_samplerate_unit(os.path.join(data_path, cell_path, 'stimuli.dat'), 1)
    md = relacs_metadata(os.path.join(data_path, cell_path, 'basespikes1.dat'), lower_keys=True, flat=True, add_sections=True)
    ds = md['duration']
    i = ds.find('s')
    duration = float(ds[:i])
    n = int((duration+0.01)*samplerate)

    # load EOD trace:
    file_name = glob.glob(os.path.join(data_path, cell_path, 'trace-2.raw*'))[0]
    if file_name[-3:] == '.gz':
        with gzip.open(file_name, 'rb') as sf:
            data = np.frombuffer(sf.read(4*n), dtype=np.float32)
    else:
        with open(file_name, 'rb') as sf:
            data = np.fromfile(sf, np.float32, count=n)

    # plot histogram or recording:
    if plot_data_trace:
        fig, ax = plt.subplots()
        ax.set_title(cell_path)
        ax.plot(data)
        plt.show()
            
    # plot histogram or recording:
    if plot_data_hist:
        fig, ax = plt.subplots()
        ax.set_title(cell_path)
        mean = np.mean(data)        
        std = np.std(data)        
        ax.axvline(mean - std, color='gray', ls=':')
        ax.axvline(mean, color='gray', ls='-')
        ax.axvline(mean + std, color='gray', ls=':')
        ax.hist(data, 200)
        plt.show()

    # detect EOD times:
    eod_times = detect_eods(data, samplerate)

    # plot detected EODs:
    if plot_detection:
        fig, ax = plt.subplots()
        time = np.arange(0, 0.5, 1/samplerate)
        ax.plot(time, data[:len(time)])
        sub_eods = eod_times[eod_times < time[-1]]
        ax.plot(sub_eods, 1.8*np.ones(len(sub_eods)), 'o')
        plt.show()        

    # report CV and max of intervals:
    if check_cv:
        ieis = np.diff(eod_times)
        mean = np.mean(ieis)
        cv = np.std(ieis)/mean
        max_eod = np.max(ieis)
        print(f'{cell_path:30} CV={cv:5.3f}  mean={1000*mean:4.1f}ms  max={1000*max_eod:4.1f}ms')

    # plot interval histogram:
    if check_iei_hist:
        fig, ax = plt.subplots()
        ax.set_title(cell_path)
        plot_eod_interval_hist(ax, eod_times)
        plt.show()
    
    # save EOD peak times:
    if save_eods:
        np.save(os.path.join(cell_path, 'baseline_eods_trial_1.npy'), eod_times)
