import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cPickle as pickle
import math
import h5py

from copy import deepcopy

caffe_root = '/home/ctnuser/saubin/src/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Open an IPython session if an exception is found
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

path_prefix = '/home/ctnuser/saubin/src/datasets/DatasetEynsham/Images/'
index_mat = sio.loadmat(path_prefix + 'IndexToFilename.mat')['IndexToFilename'][0]
testing_start_index = 4804
testing_end_index = len(index_mat)

training_images = []
testing_images = []

# AlexNet can do a batch_size of 50
# GoogLeNet needs a smaller batch_size, 10 works
batch_size = 10 #50

# Smush the 5 images together into one, otherwise treat them separately
smush = True

# Which layer to get the features from
layer = 'inception_4e/3x3' #'conv4'

if smush:
    # TODO: make sure concatenation is along the correct axis
    for i in range(testing_start_index):
        training_images.append([ index_mat[i][0,0][0],
                                 index_mat[i][0,1][0],
                                 index_mat[i][0,2][0],
                                 index_mat[i][0,3][0],
                                 index_mat[i][0,4][0],
                               ])

    for i in range(testing_start_index, testing_end_index):
        testing_images.append([ index_mat[i][0,0][0],
                                index_mat[i][0,1][0],
                                index_mat[i][0,2][0],
                                index_mat[i][0,3][0],
                                index_mat[i][0,4][0],
                              ])
else:
    for i in range(testing_start_index):
      for j in range(5):
          training_images.append(index_mat[i][0,j][0])

    for i in range(testing_start_index, testing_end_index):
      for j in range(5):
          testing_images.append(index_mat[i][0,j][0])

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
#net.blobs['data'].reshape(batch_size,3,227,227) # AlexNet uses 227*227
net.blobs['data'].reshape(batch_size,3,224,224) # GoogLeNet uses 224x224

# TODO: use something better than a list
training_features = []

confusion_matrix = np.zeros((len(training_images), len(testing_images)))

# take an array of shape (n, height, width) or (n, height, width, channels)
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.imshow(data)
    plt.show()

def smush_images(im_list):

    return np.concatenate( map(lambda x: caffe.io.load_image(path_prefix + x), im_list) )

# Get all the features for the training images
b = 0
for batch in range(int(len(training_images) / batch_size)):
  if smush:
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                      smush_images(x)),
                                      training_images[b:b+batch_size])
  else:
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                      caffe.io.load_image(path_prefix + x)),
                                      training_images[b:b+batch_size])
  out = net.forward()
  print("Training Batch %i of %i" % (batch, int(len(training_images) / batch_size)))

  for bi in range(batch_size):

    feat = net.blobs[layer].data[bi]
    #vis_square(feat, padval=0.5)

    training_features.append(deepcopy(feat))

# Run the last partial batch if needed
extra = len(training_images) % batch_size
if extra != 0:
  if smush:
      net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                      smush_images(x)),
                                      training_images[-extra:])
  else:
      net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                      caffe.io.load_image(path_prefix + x)),
                                      training_images[-extra:])
  out = net.forward()
  print("Training Overflow Batch")

  for bi in range(extra):

    feat = net.blobs[layer].data[bi]

    training_features.append(deepcopy(feat))

j = 0
for batch in range(int(len(testing_images) / batch_size)):
  if smush:
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                      smush_images(x)),
                                      testing_images[b:b+batch_size])
  else:
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                      caffe.io.load_image(path_prefix + x)),
                                      testing_images[b:b+batch_size])
  out = net.forward()
  print("Testing Batch %i of %i" % (batch, int(len(testing_images) / batch_size)))

  for bi in range(batch_size):

    feat = net.blobs[layer].data[bi]

    for i in range(len(training_images)):
      confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])
    j += 1

# Run the last partial batch if needed
extra = len(testing_images) % batch_size
if extra != 0:
  if smush:
      net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                      smush_images(x)),
                                      testing_images[-extra:])
  else:
      net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                      caffe.io.load_image(path_prefix + x)),
                                      testing_images[-extra:])
  out = net.forward()
  print("Testing Overflow Batch")

  for bi in range(extra):

    feat = net.blobs[layer].data[bi]

    for i in range(len(training_images)):
      confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])
    j += 1

#for i in range(len(training_images)):
#  vis_square(training_features[i], padval=0.5)

print( confusion_matrix )
#print( "Saving Confusion Matrix to Pickle File..." )
#pickle.dump(confusion_matrix, open('test_confusion_matrix_smush.p','wb'))
#print( "Saving Complete!" )

# Remove any slashes from layer name
layer = layer.replace('/','-')

print( "Saving Confusion Matrix for %s to HDF5 File..." % layer )
h5f = h5py.File('confusion_matrix_smush_googlenet_%s.h5' % layer, 'w')
h5f.create_dataset('dataset', data=confusion_matrix)
h5f.close()
print( "Saving Complete!" )
