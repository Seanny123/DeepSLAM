# Visualizing the Confusion Matrix

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

fname = 'conf_mat_smush_inception_4e-3x3.h5'
dname = 'dataset'

if len(sys.argv) >= 2:
  fname = sys.argv[1]
  dname = 'dataset'
  layer = 10

if len(sys.argv) == 3:
  layer = int(sys.argv[2])

h5f = h5py.File(fname, 'r')
data = h5f[dname][:]
h5f.close()

print(data)

# If multiple confusion matrices are saved, only display a specific layer
if len(data.shape) == 3:
  plt.imshow(data[layer,...])
else:
  plt.imshow(data)
plt.show()
