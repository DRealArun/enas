from __future__ import print_function

import os
import sys
#import cPickle as pickle
import pickle as pickle
import numpy as np
import tensorflow as tf
import os, sys, tarfile, urllib
import scipy as sp
import scipy.io as sio
from scipy.misc import *
import urllib.request


def _read_data(data_path, train_files):
  """Reads CIFAR-10 format data. Always returns NHWC format.

  Returns:
    images: np tensor of size [N, H, W, C]
    labels: np tensor of size [N]
  """
  images, labels = [], []
  for file_name in train_files:
    print(file_name)
    full_name = os.path.join(data_path, file_name)
    with open(full_name,'rb') as finp:
      data = pickle.load(finp,encoding='latin1')
      batch_images = data["data"].astype(np.float32) / 255.0
      batch_labels = np.array(data["labels"], dtype=np.int32)
      images.append(batch_images)
      labels.append(batch_labels)
  images = np.concatenate(images, axis=0)
  labels = np.concatenate(labels, axis=0)
  images = np.reshape(images, [-1, 3, 32, 32])
  images = np.transpose(images, [0, 2, 3, 1])

  return images, labels


def read_cifar_data(data_path, num_valids=5000):
  print("-" * 80)
  print("Reading data")

  images, labels = {}, {}

  train_files = [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
  ]
  test_file = [
    "test_batch",
  ]
  images["train"], labels["train"] = _read_data(data_path, train_files)

  if num_valids:
    images["valid"] = images["train"][-num_valids:]
    labels["valid"] = labels["train"][-num_valids:]

    images["train"] = images["train"][:-num_valids]
    labels["train"] = labels["train"][:-num_valids]
  else:
    images["valid"], labels["valid"] = None, None

  images["test"], labels["test"] = _read_data(data_path, test_file)

  print("Prepropcess: [subtract mean], [divide std]")
  mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
  std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

  print("mean: {}".format(np.reshape(mean * 255.0, [-1])))
  print("std: {}".format(np.reshape(std * 255.0, [-1])))

  images["train"] = (images["train"] - mean) / std
  if num_valids:
    images["valid"] = (images["valid"] - mean) / std
  images["test"] = (images["test"] - mean) / std

  return images, labels


def read_mnist_data(data_path, num_valids=8000):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_path, one_hot=False, reshape=False, validation_size= num_valids)
    images, labels = {}, {}
    def _extract_fn(x):
        X = x.images
        y = np.array(x.labels, dtype=np.int32)

#         if not normalize_range:
#             X *= 255.0
        X = np.reshape(X, [-1, 1, 28, 28])
        X = np.transpose(X, [0, 2, 3, 1])
        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    if num_valids:
        images["valid"] = Xval
        labels["valid"] = yval

        images["train"] = Xtrain
        labels["train"] = ytrain
    else:
        images["valid"], labels["valid"] = None, None

    images["test"], labels["test"] = Xtest, ytest
    print("Max value before preprocessing",np.max(images["train"][0]))
    print ("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print ("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
        images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std
    print("Max value after preprocessing",np.max(images["train"][0]))
    return images, labels


def read_fashion_data(data_path, num_valids=5000):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_path, one_hot=False, reshape=False, validation_size= num_valids)
    images, labels = {}, {}
    def _extract_fn(x):
        X = x.images
        y = np.array(x.labels, dtype=np.int32)

#         if not normalize_range:
#             X *= 255.0
        X = np.reshape(X, [-1, 1, 28, 28])
        X = np.transpose(X, [0, 2, 3, 1])
        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    if num_valids:
        images["valid"] = Xval
        labels["valid"] = yval

        images["train"] = Xtrain
        labels["train"] = ytrain
    else:
        images["valid"], labels["valid"] = None, None

    images["test"], labels["test"] = Xtest, ytest
    print("Max value before preprocessing",np.max(images["train"][0]))
    print ("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print ("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
        images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std
    print("Max value after preprocessing",np.max(images["train"][0]))
    return images, labels


# Below lies the code to read in the STL10 dataset and it is taken in parts from,
# https://github.com/ltoscano/STL10/blob/master/stl10_input.py
def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        labels = np.array(labels, dtype=np.int32)
        return labels


def read_all_images(path_to_data):
    with open(path_to_data, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def download_and_extract(DATA_DIR):
    DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        # def _progress(count, block_size, total_size):
        #     sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
        #         float(count * block_size) / float(total_size) * 100.0))
        #     sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_stl10_data(data_path, num_valids=500):
    images, labels = {}, {}
    # path to the binary train file with image data
    train_img_path = os.path.join(data_path,'stl10_binary','train_X.bin')

    # path to the binary train file with labels
    train_label_path = os.path.join(data_path,'stl10_binary','train_y.bin')

    # path to the binary test file with image data
    test_img_path = os.path.join(data_path,'stl10_binary','test_X.bin')

    # path to the binary test file with labels
    test_label_path = os.path.join(data_path,'stl10_binary','test_y.bin')

    download_and_extract(data_path)

    # test to check if the whole dataset is read correctly
    images_train = read_all_images(train_img_path)
    print("Training images",images_train.shape)

    labels_train = read_labels(train_label_path)
    print("Training labels",labels_train.shape)

    images_test = read_all_images(test_img_path)
    print("Test images",images_test.shape)

    labels_test = read_labels(test_label_path)
    print("Test labels",labels_test.shape)

    images["train"] = images_train.astype(np.float32) / 255.0
    labels["train"] = labels_train

    if num_valids:
        images["valid"] = images["train"][-num_valids:]
        labels["valid"] = labels["train"][-num_valids:]

        images["train"] = images["train"][:-num_valids]
        labels["train"] = labels["train"][:-num_valids]
    else:
        images["valid"], labels["valid"] = None, None

    images["test"], labels["test"] = images_test.astype(np.float32) / 255.0, labels_test
    print("Max value before preprocessing",np.max(images["train"][0]))
    print ("Prepropcess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print ("mean: {}".format(np.reshape(mean * 255.0, [-1])))
    print ("std: {}".format(np.reshape(std * 255.0, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
        images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std
    print("Max value after preprocessing",np.max(images["train"][0]))
    return images, labels

# STL10 dataset code ends above

# Below lies the code for SVHN
# Parts taken from https://github.com/codemukul95/SVHN-classification-using-Tensorflow/blob/master/load_input.py
def read_svhn_data(data_path, num_valids = 10000):
    images, labels = {}, {}
    train_path = os.path.join(data_path, 'train_32x32')
    train_dict = sio.loadmat(train_path)
    X = np.asarray(train_dict['X'], dtype=np.float32)

    X_train = []
    for i in range(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in range(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0

    images["train"] = X_train
    labels["train"] = np.squeeze(Y_train)
    labels["train"] = np.array(labels["train"], dtype=np.int32)

    train_path = os.path.join(data_path, 'test_32x32')
    test_dict = sio.loadmat(train_path)
    X = np.asarray(test_dict['X'], dtype=np.float32)

    X_test = []
    for i in range(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in range(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0

    images["test"] = X_test
    labels["test"] = np.squeeze(Y_test)
    labels["test"] = np.array(labels["test"], dtype=np.int32)

    if num_valids:
        images["valid"] = images["train"][-num_valids:]
        labels["valid"] = labels["train"][-num_valids:]

        images["train"] = images["train"][:-num_valids]
        labels["train"] = labels["train"][:-num_valids]
    else:
        images["valid"], labels["valid"] = None, None

    print("Max value before preprocessing",np.max(images["train"][0]))
    print ("Preprocess: [subtract mean], [divide std]")
    mean = np.mean(images["train"], axis=(0, 1, 2), keepdims=True)
    std = np.std(images["train"], axis=(0, 1, 2), keepdims=True)

    print ("mean: {}".format(np.reshape(mean, [-1])))
    print ("std: {}".format(np.reshape(std, [-1])))

    images["train"] = (images["train"] - mean) / std
    if num_valids:
        images["valid"] = (images["valid"] - mean) / std
    images["test"] = (images["test"] - mean) / std
    print("Max value after preprocessing",np.max(images["train"][0]))
    return images, labels


def read_data(data_path, dataset, num_valids=5000):
  images, labels = {}, {}
  if dataset == 'cifar10':
    images, labels = read_cifar_data(data_path, num_valids)
  elif dataset == 'mnist':
    images, labels = read_mnist_data(data_path, num_valids)
  elif dataset == 'fashion':
    images, labels = read_fashion_data(data_path, num_valids)
  elif dataset == 'svhn':
    images, labels = read_svhn_data(data_path, num_valids)
  elif dataset == 'stl10':
    images, labels = read_stl10_data(data_path, num_valids)
  # elif dataset == 'devanagari':
  #   images, labels = read_devanagari_data(data_path, num_valids)
  else:
    assert False, "Dataset not supported"
  return images, labels
