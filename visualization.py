# Visualizing the Confusion Matrix

import h5py
import matplotlib.pyplot as plt
import sys
import seaborn as sns

sns.set_style('white')

layer = 0
fname = 'conf_mat_smush_inception_4e-3x3.h5'
dname = 'dataset'

if len(sys.argv) >= 2:
    fname = sys.argv[1]
    dname = 'dataset'
    layer = 10

if len(sys.argv) == 3:
    layer = int(sys.argv[2])

with h5py.File(fname, 'r') as h5f:
    # data = h5f[dname][:] # Everything
    if ('conf_mat' in fname) and ('full' not in fname):
        data = h5f[dname][:]  # Only the train vs test data
    else:
        data = h5f[dname][0:4789, 4789:9575]  # Only the train vs test data

print(data)

# If multiple confusion matrices are saved, only display a specific layer
if len(data.shape) == 3:
    plt.imshow(data[layer, ...])
else:
    plt.imshow(data)

plt.title("Confusion Matrix")
#plt.set_xlabel("Training Index")
#plt.set_ylabel("Testing Index")

plt.show()
