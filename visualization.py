# Visualizing the Confusion Matrix

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys

fname = 'conf_mat_smush_inception_4e-3x3.h5'
dname = 'dataset'

if len(sys.argv) == 2:
  fname = sys.argv[1]
  dname = 'dataset'

h5f = h5py.File(fname, 'r')
data = h5f[dname][:]
h5f.close()

print(data)

plt.imshow(data)
plt.show()
