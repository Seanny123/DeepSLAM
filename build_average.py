# make conf_mat files with averages
import numpy as np
import h5py
import sys
from copy import deepcopy

dname = 'dataset'

prefix = 'conf_mat_smush_full_googlenet_inception_'
fnames = ['3a-output.h5', '3b-output.h5',
          '4a-output.h5', '4b-output.h5',
          '4c-output.h5', '4d-output.h5',
          '4e-output.h5',
          '5a-output.h5', '5b-output.h5'
         ]

data = None
data_all = None
data_net = None
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  if data is None:
    data = h5f[dname][:]
    data_all = h5f[dname][:]
  else:
    data += h5f[dname][:]
    data_all += h5f[dname][:]
  h5f.close()

prefix = 'conf_mat_smush_googlenet_'
fnames = ['pool1-norm1.h5', 'conv2-norm2.h5']

for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  data_all[0:4789,4789:9575] += h5f[dname][:]
  data[0:4789,4789:9575] += h5f[dname][:]
  h5f.close()

data_net = data/11.0
h5f = h5py.File('googlenet_avg.h5', 'w')
h5f.create_dataset('dataset', data=data/11.0)
h5f.close()

data = None

prefix = 'conf_mat_smush_full_overfeat_'
fnames = [
          '0.h5', '2.h5', '3.h5', #TODO: 1 layer missing
          '4.h5', '5.h5',
          '6.h5', '7.h5',
          '8.h5', '9.h5',
          '10.h5', '11.h5',
          '10.h5', '11.h5',
          '12.h5', '13.h5',
          '14.h5', '15.h5',
          '16.h5', '17.h5',
         ]
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  if data is None:
    data = h5f[dname][:]
    data_all += h5f[dname][:]
  else:
    data += h5f[dname][:]
    data_all += h5f[dname][:]
  h5f.close()

data_net += data/21.0
h5f = h5py.File('overfeat_avg.h5', 'w')
h5f.create_dataset('dataset', data=data/21.0)
h5f.close()

data = None

prefix = 'conf_mat_smush_full_caffenet_'
fnames = ['conv3.h5', 'conv4.h5',
         ]
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  if data is None:
    data = h5f[dname][:]
    data_all += h5f[dname][:]
  else:
    data += h5f[dname][:]
    data_all += h5f[dname][:]
  h5f.close()
prefix = 'conf_mat_smush_caffenet_'
fnames = ['conv1.h5', 'conv1.h5',
         ]
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  data_all[0:4789,4789:9575] += h5f[dname][:]
  data[0:4789,4789:9575] += h5f[dname][:]
  h5f.close()

data_net += data/4.0
h5f = h5py.File('alexnet_avg.h5', 'w')
h5f.create_dataset('dataset', data=data/4.0)
h5f.close()

data = None

prefix = 'conf_mat_smush_full_vgg19_'
fnames = ['conv4_4.h5',
          'conv5_3.h5', 'conv5_4.h5',
         ]
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  if data is None:
    data = h5f[dname][:]
    data_all += h5f[dname][:]
  else:
    data += h5f[dname][:]
    data_all += h5f[dname][:]
  h5f.close()

data_net += data/2.0
h5f = h5py.File('vgg19_avg.h5', 'w')
h5f.create_dataset('dataset', data=data/2.0)
h5f.close()

# Without Cifar10 averages
h5f = h5py.File('all_net_avg_no_cifar.h5', 'w')
h5f.create_dataset('dataset', data=data_net/4.0)
h5f.close()

h5f = h5py.File('all_layer_avg_no_cifar.h5', 'w')
h5f.create_dataset('dataset', data=data_all/33.0)
h5f.close()
data = None

prefix = 'conf_mat_smush_full_cifar10full_'
fnames = ['conv1.h5', 'conv2.h5', 'conv3.h5',
         ]
for fname in fnames:
  print(prefix + fname)
  h5f = h5py.File(prefix + fname, 'r')
  if data is None:
    data = h5f[dname][:]
    data_all += h5f[dname][:]
  else:
    data += h5f[dname][:]
    data_all += h5f[dname][:]
  h5f.close()

data_net += data/3.0
h5f = h5py.File('cifar10_avg.h5', 'w')
h5f.create_dataset('dataset', data=data/3.0)
h5f.close()

print(data)

h5f = h5py.File('all_net_avg.h5', 'w')
h5f.create_dataset('dataset', data=data_net/5.0)
h5f.close()

h5f = h5py.File('all_layer_avg.h5', 'w')
h5f.create_dataset('dataset', data=data_all/36.0)
h5f.close()

# TODO: do one looking at averaging only the top 2 layers of each net
# (and skipping Cifar10)
