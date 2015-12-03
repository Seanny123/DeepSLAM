import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cPickle as pickle
import math
import h5py
import getpass
import sys
import overfeat
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
elif user == 'saubin': #TODO: put in Sean's actual path, I just guessed for now
  caffe_root = '/home/saubin/src/caffe/'
  overfeat_root = '/home/saubin/src/OverFeat/'
  path_prefix = '/home/saubin/src/datasets/DatasetEynsham/Images/'
else:
  caffe_root = '/home/ctnuser/saubin/src/caffe/'
  overfeat_root = '/home/ctnuser/saubin/src/OverFeat/'
  path_prefix = '/home/ctnuser/saubin/src/datasets/DatasetEynsham/Images/'

sys.path.insert(0, caffe_root + 'python')

import caffe

# Open an IPython session if an exception is found
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

# Stuff for optional plotting
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

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

# TODO: use something better than a list
training_features = []

# OverFeat does not use caffe
if net_type == 'OverFeat':
  
  # OverFeat has 22 layers, including original image
  num_layers = 22
  
  # For filename purposes
  layer = 'all'
  layer = 10
  
  if layer == 'all':
    # Put all layers into one stacked confusion matrix
    confusion_matrix = np.zeros((num_layers, len(training_images), len(testing_images)))
  else:
    # Make the confusion matrix for a single layer
    confusion_matrix = np.zeros((len(training_images), len(testing_images)))

  overfeat.init(overfeat_root + 'data/default/net_weight_0', 0)
  
  for i in range(len(training_images)):

    print("Training Image %s of %s" % (i, len(training_images)))

    if smush:
      image = smush_overfeat_images(training_images[i])
    else:
      image = load_overfeat_image(training_images[i])

    b = overfeat.fprop(image)
    
    if layer == 'all':
      # Calculate features for all layers at once
      features = []
      for n in range(num_layers):
        features.append(deepcopy(overfeat.get_output(n)))

      training_features.append(features)
    else:
      training_features.append(deepcopy(overfeat.get_output(layer)))
  
  for i in range(len(testing_images)):

    print("Testing Image %s of %s" % (i, len(testing_images)))

    if smush:
      image = smush_overfeat_images(testing_images[i])
    else:
      image = load_overfeat_image(testing_images[i])

    b = overfeat.fprop(image)

    for j in range(len(training_images)):
      if layer == 'all':
        for n in range(num_layers):
          feat = overfeat.get_output(n)
        
          confusion_matrix[n,j,i] = np.linalg.norm(feat - training_features[j][n])
      else:
        feat = overfeat.get_output(layer)
      
        confusion_matrix[j,i] = np.linalg.norm(feat - training_features[j])

  # Convert to string in case it is a layer number, for use in the filename
  layer = str(layer)

# Use caffe for all other models
else:
  confusion_matrix = np.zeros((len(training_images), len(testing_images)))
  
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
    batch_size = 10
    layer = 'prob'
    #layer = 'inception_3a/output'
    layer = 'inception_3b/output'
    #layer = 'inception_4a/output'
    layer = 'inception_4b/output'
    layer = 'inception_4c/output'
    layer = 'inception_4d/output'
    layer = 'inception_4e/output'
    #layer = 'inception_5a/output'
    layer = 'inception_5b/output'
    net.blobs['data'].reshape(batch_size,3,224,224) # GoogLeNet uses 224x224
  elif net_type == 'AlexNet' or net_type == 'CaffeNet':
    batch_size = 50
    layer = 'conv3'
    net.blobs['data'].reshape(batch_size,3,227,227) # AlexNet uses 227*227
  if 'Cifar10' in net_type:
    batch_size = 10
    layer = 'conv1'
    net.blobs['data'].reshape(batch_size,3,32,32) # Cifar10Net uses 32x32

  # Get all the features for the training images
  for batch in range(int(len(training_images) / batch_size)):
    if smush:
      net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                        smush_images(x)),
                                        training_images[batch*batch_size:(batch+1)*batch_size])
    else:
      net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                        caffe.io.load_image(path_prefix + x)),
                                        training_images[batch*batch_size:(batch+1)*batch_size])
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
                                        testing_images[batch*batch_size:(batch+1)*batch_size])
    else:
      net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data',
                                        caffe.io.load_image(path_prefix + x)),
                                        testing_images[batch*batch_size:(batch+1)*batch_size])
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

  # Remove any slashes from layer name
  layer = layer.replace('/','-')



# Optional plotting of features
#for i in range(len(training_images)):
#  vis_square(training_features[i], padval=0.5)
#plt.show()

print( confusion_matrix )

# Construct file name
fname = 'conf_mat'
if smush:
  fname += '_smush'
if full:
  fname += '_full'
fname += '_' + net_type.lower() + '_' + layer + '.h5'

# Save to HDF5 format
print( "Saving Confusion Matrix for %s to HDF5 File..." % layer )
h5f = h5py.File(fname, 'w')
h5f.create_dataset('dataset', data=confusion_matrix)
h5f.close()
print( "Saving Complete!" )
