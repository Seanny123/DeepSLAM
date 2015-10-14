import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

caffe_root = '/home/bjkomer/caffe/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe.set_mode_cpu()
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
net.blobs['data'].reshape(50,3,227,227)

fileroot = '/home/bjkomer/Pictures/Textures/'
filenames = ['Aircos0028_S.jpg', 'BrickLargeBare0124_7_S.jpg',
             'BrickLargeBrown0017_2_S.jpg', 'BrickRound0046_2_S.jpg',
             'BrickRound0098_7_S.jpg']

# just use the same for debugging for now
training_images = filenames
testing_images = filenames

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
for filename in training_images:
  net.blobs['data'].data[...] = transformer.preprocess('data',
                                                       caffe.io.load_image(fileroot + filename))
  out = net.forward()
  #print("Predicted class is #{}.".format(out['prob'].argmax()))

  feat = net.blobs['conv4'].data[0]

  print(net.blobs['conv4'].data.shape)

  training_features.append(deepcopy(feat))

# Get all the features for the testing images and compare to eat training image
for j, filename in enumerate(testing_images):

  net.blobs['data'].data[...] = transformer.preprocess('data',
                                                       caffe.io.load_image(fileroot + filename))
  out = net.forward()
  #print("Predicted class is #{}.".format(out['prob'].argmax()))

  feat = net.blobs['conv4'].data[0]

  

  #vis_square(feat, padval=0.5)
  for i in range(len(training_images)):
    #vis_square(feat-training_features[i], padval=0.5)
    confusion_matrix[i,j] = np.linalg.norm(feat*1e9 - training_features[i]*1e9)


for i in range(len(training_images)):
  vis_square(training_features[i], padval=0.5)

print( confusion_matrix )
print( confusion_matrix[0,1] )
