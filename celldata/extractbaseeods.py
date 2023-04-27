import os
import glob
import gzip
import numpy as np
import matplotlib.pyplot as plt
from thunderfish.eventdetection import detect_peaks, std_threshold

#2010-11-08-al-invivo-1/trace-2.raw.gz

data_path = ''
cell_path = ''
file_name = 'trace-2.raw.gz'
#file_name = 'trace-2.raw'

samplerate = 20000
n = 100000

# load EOD trace:
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
if not os.path.isdir(cell_path):
    os.mkdir(cell_path)
np.save(os.path.join(cell_path, 'baseline_eods_trial_1.npy'), eod_times)

