import os
import glob
import gzip
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.eventdetection import detect_peaks, std_threshold
from thunderfish.dataloader import relacs_samplerate_unit, relacs_metadata

data_path = 'data'

for cell_path in sorted(glob.glob('2*')):
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

    # detect peaks:
    thresh = std_threshold(data, thresh_fac=3)
    p, _ = detect_peaks(data, thresh)
    eod_times = p/samplerate

    # save EOD peak times:
    np.save(os.path.join(cell_path, 'baseline_eods_trial_1.npy'), eod_times)
