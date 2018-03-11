import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cPickle as pickle
import math
import h5py
import getpass
import sys
import overfeat
import time
from scipy.ndimage import imread
from scipy.misc import imresize
from copy import deepcopy

# Smush the 5 images together into one, otherwise treat them separately
smush = True

# Create the full confusion matrix, including sections not needed
full = True

# Whether or not the images have a colour channel
colour = False

# The type of pre-trained deep network to get the features from
net_type = 'GoogLeNet'
#net_type = 'AlexNet'
#net_type = 'CaffeNet'
#net_type = 'OverFeat'
#anet_type = 'Cifar10'
#net_type = 'Cifar10Full'
#net_type = 'Cifar10SoftLIF'

# Check the username, so the same code can work on all of our computers
user = getpass.getuser()
if user == 'ctnuser':
  caffe_root = '/home/ctnuser/saubin/src/caffe/'
  overfeat_root = '/home/ctnuser/saubin/src/OverFeat/'
  path_prefix = '/home/ctnuser/saubin/src/datasets/DatasetEynsham/Images/'
elif user == 'bjkomer':
  caffe_root = '/home/bjkomer/caffe/'
  overfeat_root = '/home/bjkomer/OverFeat/'
  path_prefix = '/home/bjkomer/deep_learning/datasets/DatasetEynsham/Images/'
else:
  caffe_root = '/home/ctnuser/saubin/src/caffe/'
  overfeat_root = '/home/ctnuser/saubin/src/OverFeat/'
  path_prefix = '/home/ctnuser/saubin/src/datasets/DatasetEynsham/Images/'

sys.path.insert(0, caffe_root + 'python')

import caffe

# Stuff for optional plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def vis_square(data, padsize=1, padval=0):
    """
    take an array of shape (n, height, width) or (n, height, width, channels)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """

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
    plt.figure()
    #plt.show()


def smush_images(im_list):

    return np.concatenate( map(lambda x: caffe.io.load_image(path_prefix + x), im_list) )


def process_overfeat_image(image):

    # resize and crop into a 231x231 image
    h0 = image.shape[0]
    w0 = image.shape[1]
    d0 = float(min(h0, w0))

    # TODO: make this less hacky and more legit (if possible)
    if not colour:
      # Copy the monochrome image to all three channels to make OverFeat happy
      image = image.reshape(h0,w0,1)
      image = np.concatenate([image, image, image], axis=2)

    image = image[int(round((h0-d0)/2.)):int(round((h0-d0)/2.)+d0),
                  int(round((w0-d0)/2.)):int(round((w0-d0)/2.)+d0), :]
    image = imresize(image, (231, 231)).astype(np.float32)

    # numpy loads image with colors as last dimension, transpose tensor
    h = image.shape[0]
    w = image.shape[1]
    c = image.shape[2]
    image = image.reshape(w*h, c)
    image = image.transpose()
    image = image.reshape(c, h, w)

    return image


def load_overfeat_image(im):
    # read image
    return process_overfeat_image(imread(path_prefix + im))


def smush_overfeat_images(im_list):

    return process_overfeat_image(np.concatenate( map(lambda x:
                                                      imread(path_prefix + x),
                                                      im_list) ))


index_mat = sio.loadmat(path_prefix + 'IndexToFilename.mat')['IndexToFilename'][0]

# MATLAB code uses 4789 as the split point, and this seems to match the data beter
# The dataset itself claims 4804 is the split point, but this looks to be incorrect
if full:
  training_start_index = 0
  training_end_index = len(index_mat)
  testing_start_index = 0
  testing_end_index = len(index_mat)
else:
  training_start_index = 0
  training_end_index = 4789 #4804
  testing_start_index = 4789 #4804
  testing_end_index = len(index_mat)

training_images = []
testing_images = []

if smush:
    # TODO: make sure concatenation is along the correct axis
    for i in range(training_start_index, training_end_index):
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
    for i in range(training_start_index, training_end_index):
      for j in range(5):
          training_images.append(index_mat[i][0,j][0])

    for i in range(testing_start_index, testing_end_index):
      for j in range(5):
          testing_images.append(index_mat[i][0,j][0])

#net_types = ["OverFeat", "GoogLeNet", "CaffeNet", "Cifar10"]
net_types = ["GoogLeNet", "CaffeNet", "Cifar10"]
num_images = [10,50,100,500,1000]
timing_data = {}

for net_type in net_types:
    timing_data[net_type] = {}
    for num_image in num_images:
        print("Timing %s for %s images" % (net_type, num_image))

        # OverFeat does not use caffe
        if net_type == 'OverFeat':

          overfeat.init(overfeat_root + 'data/default/net_weight_0', 0)

          start = time.clock()
          for i in range(num_image):

            image = smush_overfeat_images(training_images[i])

            b = overfeat.fprop(image)

          end = time.clock()
          elapsed = end - start
          print(elapsed)
          timing_data[net_type][num_image] = elapsed

        # Use caffe for all other models
        else:

          if user == 'ctnuser':
            caffe.set_mode_gpu()
          else:
            caffe.set_mode_cpu()


          if net_type == 'GoogLeNet':
            net = caffe.Net(caffe_root + 'models/bvlc_googlenet/deploy.prototxt',
                            caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel',
                            caffe.TEST)
          elif net_type == 'CaffeNet':
            net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                            caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                            caffe.TEST)
          elif net_type == 'AlexNet':
            net = caffe.Net(caffe_root + 'models/bvlc_alexnet/deploy.prototxt',
                            caffe_root + 'models/bvlc_alexnet/bvlc_alexnet.caffemodel',
                            caffe.TEST)
          elif net_type == 'Cifar10':
            net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick.prototxt',
                            caffe_root + 'examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5',
                            caffe.TEST)
          elif net_type == 'Cifar10Full':
            net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_full.prototxt',
                            caffe_root + 'examples/cifar10/cifar10_full_iter_70000.caffemodel.h5',
                            caffe.TEST)
          elif net_type == 'Cifar10SoftLIF':
            net = caffe.Net(caffe_root + 'examples/cifar10/cifar10_quick_softlif.prototxt',
                            caffe_root + 'examples/cifar10/cifar10_quick_softlif_iter_5000.caffemodel.h5',
                            caffe.TEST)

          # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
          transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
          transformer.set_transpose('data', (2,0,1))
          transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
          transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
          transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

          # AlexNet can do a batch_size of 50
          # GoogLeNet needs a smaller batch_size, 10 works
          # They also have different names for each layer
          if net_type == 'GoogLeNet':
            batch_size = 1#10
            net.blobs['data'].reshape(batch_size,3,224,224) # GoogLeNet uses 224x224
          elif net_type == 'AlexNet' or net_type == 'CaffeNet':
            batch_size = 1#50
            net.blobs['data'].reshape(batch_size,3,227,227) # AlexNet uses 227*227
          if 'Cifar10' in net_type:
            batch_size = 1#10
            net.blobs['data'].reshape(batch_size,3,32,32) # Cifar10Net uses 32x32

          start = time.clock()

          # Get all the features for the training images
          for batch in range(int(num_image / batch_size)):
              net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                                smush_images(x)),
                                                training_images[batch*batch_size:(batch+1)*batch_size])
              out = net.forward()

          # Run the last partial batch if needed
          extra = num_image % batch_size
          if extra != 0:
              net.blobs['data'].data[:extra,...] = map(lambda x: transformer.preprocess('data',
                                                       smush_images(x)),
                                                       training_images[-extra:])
              out = net.forward()

          end = time.clock()
          elapsed = end - start
          print(elapsed)
          timing_data[net_type][num_image] = elapsed


print(timing_data)

# Construct file name
fname = 'timing_data.p'

# Save to HDF5 format
print("Saving Timing Data to Pickle File...")
pickle.dump(timing_data, open(fname, 'wb'))
print("Saving Complete!")
