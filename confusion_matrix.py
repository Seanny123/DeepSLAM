import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cPickle as pickle
import math

from copy import deepcopy

caffe_root = '/home/ctnuser/saubin/src/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

path_prefix = '/home/ctnuser/saubin/src/datasets/DatasetEynsham/Images/'
index_mat = sio.loadmat(path_prefix + 'IndexToFilename.mat')['IndexToFilename'][0]
testing_start_index = 4804
testing_end_index = len(index_mat)

training_images = []
testing_images = []
batch_size = 50

for i in range(testing_start_index):
  #TODO Smush images from the set of 5 together
  for j in range(5):
      training_images.append(index_mat[i][0,j][0])

for i in range(testing_start_index, testing_end_index):
  #TODO Smush images from the set of 5 together
  for j in range(5):
      testing_images.append(index_mat[i][0,j][0])

caffe.set_mode_gpu()
net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 50
net.blobs['data'].reshape(batch_size,3,227,227)

#fileroot = '/home/bjkomer/Pictures/Textures/'
#filenames = ['Aircos0028_S.jpg', 'BrickLargeBare0124_7_S.jpg',
#             'BrickLargeBrown0017_2_S.jpg', 'BrickRound0046_2_S.jpg',
#             'BrickRound0098_7_S.jpg']

# just use the same for debugging for now
#training_images = filenames
#testing_images = filenames

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

# Get all the features for the training images
b = 0
for batch in range(int(len(training_images) / batch_size)):
#for filename in training_images:
  net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                    caffe.io.load_image(path_prefix + x)),
                                    training_images[b:b+batch_size])
  out = net.forward()
  print("Training Batch %i of %i" % (batch, int(len(testing_images) / batch_size)))

  for bi in range(batch_size):

    feat = net.blobs['conv4'].data[bi]
    #vis_square(feat, padval=0.5)

    training_features.append(deepcopy(feat))

# Run the last partial batch if needed
extra = len(training_images) % batch_size
if extra != 0:
  #net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
  net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                    caffe.io.load_image(path_prefix + x)),
                                    training_images[-extra:])
  out = net.forward()
  print("Training Overflow Batch")

  for bi in range(extra):

    feat = net.blobs['conv4'].data[bi]

    training_features.append(deepcopy(feat))

j = 0
for batch in range(int(len(testing_images) / batch_size)):
#for filename in training_images:
  net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                    caffe.io.load_image(path_prefix + x)),
                                    testing_images[b:b+batch_size])
  out = net.forward()
  print("Testing Batch %i of %i" % (batch, int(len(testing_images) / batch_size)))

  for bi in range(batch_size):

    feat = net.blobs['conv4'].data[bi]

    for i in range(len(training_images)):
      confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])
    j += 1

# Run the last partial batch if needed
extra = len(testing_images) % batch_size
if extra != 0:
  #net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
  net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                    caffe.io.load_image(path_prefix + x)),
                                    testing_images[-extra:])
  out = net.forward()
  print("Testing Overflow Batch")

  for bi in range(extra):

    feat = net.blobs['conv4'].data[bi]

    for i in range(len(training_images)):
      confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])
    j += 1


"""
# Get all the features for the testing images and compare to each training image
for j, filename in enumerate(testing_images):

  print("Testing Image: %i" % j)

  net.blobs['data'].data[b,...] = transformer.preprocess('data',
                                                       caffe.io.load_image(path_prefix + filename))
  # Increment batch index
  b += 1

  # If batch is full, run the batch
  if b == batch_size:
      b = 0
      out = net.forward()

      for bi in range(batch_size):

        feat = net.blobs['conv4'].data[bi]

        for i in range(len(training_images)):
          confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])

# Run the last partial batch if needed
if b != 0:
  out = net.forward()
  for bi in range(b):

    feat = net.blobs['conv4'].data[bi]
    
    for i in range(len(training_images)):
      confusion_matrix[i,j] = np.linalg.norm(feat - training_features[i])
"""


#for i in range(len(training_images)):
#  vis_square(training_features[i], padval=0.5)

print( confusion_matrix )

pickle.dump(confusion_matrix, open('test_confusion_matrix.p','wb'))
#pickle.dump(confusion_matrix, open('confusion_matrix.p','wb'))
