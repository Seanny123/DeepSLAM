# Visualizing averaging of multiple Confusion Matrices

import h5py
import matplotlib.pyplot as plt

fname = 'conf_mat_smush_inception_4e-3x3.h5'
dname = 'dataset'
layer = 1

prefix = 'conf_mat_smush_full_googlenet_inception_'
fnames = ['3a-output.h5', '3b-output.h5',
          '4a-output.h5', '4b-output.h5',
          '4c-output.h5', '4d-output.h5',
          '4e-output.h5',
          '5a-output.h5', '5b-output.h5'
         ]

data = None
for fname in fnames:
    print(prefix + fname)
    h5f = h5py.File(prefix + fname, 'r')
    if data is None:
        data = h5f[dname][:]
    else:
        data += h5f[dname][:]
    h5f.close()

prefix = 'conf_mat_smush_full_overfeat_'
fnames = ['10.h5', '11.h5',
          '12.h5', '13.h5',
          '14.h5', '15.h5',
          '16.h5', '17.h5',
          ]
for fname in fnames:
    print(prefix + fname)
    h5f = h5py.File(prefix + fname, 'r')
    if data is None:
        data = h5f[dname][:]
    else:
        data += h5f[dname][:]
    h5f.close()

prefix = 'conf_mat_smush_full_caffenet_'
fnames = ['conv3.h5', 'conv4.h5',
          ]
for fname in fnames:
    print(prefix + fname)
    h5f = h5py.File(prefix + fname, 'r')
    if data is None:
        data = h5f[dname][:]
    else:
        data += h5f[dname][:]
    h5f.close()

prefix = 'conf_mat_smush_full_vgg19_'
fnames = ['conv4_4.h5', 'conv5_4.h5',
          ]
for fname in fnames:
    print(prefix + fname)
    h5f = h5py.File(prefix + fname, 'r')
    if data is None:
        data = h5f[dname][:]
    else:
        data += h5f[dname][:]
    h5f.close()

print(data)

h5f = h5py.File('conf_mat_sum.h5', 'w')
h5f.create_dataset('dataset', data=data)
h5f.close()

h5f = h5py.File('conf_mat_avg.h5', 'w')
h5f.create_dataset('dataset', data=data / 21.0)
h5f.close()

# If multiple confusion matrices are saved, only display a specific layer
if len(data.shape) == 3:
    plt.imshow(data[layer, ...])
else:
    plt.imshow(data)
plt.show()
