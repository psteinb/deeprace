# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the CIFAR-10 dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

import tensorflow as tf

from deeprace.models.tf_details import resnet_model
from deeprace.models.tf_details import resnet_run_loop

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

_NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}


###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
    """Returns a list of filenames."""
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')

    assert os.path.exists(data_dir), (
        'Run cifar10_download_and_extract.py first to download and extract the '
        'CIFAR-10 data.')

    if is_training:
        return [
            os.path.join(data_dir, 'data_batch_%d.bin' % i)
            for i in range(1, _NUM_DATA_FILES + 1)
        ]
    else:
        return [os.path.join(data_dir, 'test_batch.bin')]


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_image_with_crop_or_pad(
            image, _HEIGHT + 8, _WIDTH + 8)

        # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def parse_record(raw_record, is_training):
    """Parse CIFAR-10 image and label from a raw record."""
    # Convert bytes to a vector of uint8 that is record_bytes long.
    record_vector = tf.decode_raw(raw_record, tf.uint8)

    # The first byte represents the label, which we convert from uint8 to int32
    # and then to one-hot.
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                             [_NUM_CHANNELS, _HEIGHT, _WIDTH])

    # Convert from [depth, height, width] to [height, width, depth], and cast as
    # float32.
    image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

    image = preprocess_image(image, is_training)

    return {'img': image, 'lbl': label}


################################################################################
# Functions for input processing.
################################################################################
def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_parallel_calls=1,
                           examples_per_epoch=0, multi_gpu=False):
    """Given a Dataset with raw records, parse each record into images and labels,
    and return an iterator over the records.

    Args:
      dataset: A Dataset representing raw records
      is_training: A boolean denoting whether the input is for training.
      batch_size: The number of samples per batch.
      shuffle_buffer: The buffer size to use when shuffling records. A larger
        value results in better randomness, but smaller values reduce startup
        time and use less memory.
      parse_record_fn: A function that takes a raw record and returns the
        corresponding (image, label) pair.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      examples_per_epoch: The number of examples in the current set that
        are processed each epoch. Note that this is only used for multi-GPU mode,
        and only to handle what will eventually be handled inside of Estimator.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers (see below), and can be removed
        when that is handled directly by Estimator.

    Returns:
      Dataset of (image, label) pairs ready for iteration.
    """
    # We prefetch a batch at a time, This can help smooth out the time taken to
    # load input files as we go through shuffling and processing.
    dataset = dataset.prefetch(buffer_size=batch_size)
    if is_training:
        # Shuffle the records. Note that we shuffle before repeating to ensure
        # that the shuffling respects epoch boundaries.
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)

    # If we are training over multiple epochs before evaluating, repeat the
    # dataset for the appropriate number of epochs.
    dataset = dataset.repeat(num_epochs)

    # Currently, if we are using multiple GPUs, we can't pass in uneven batches.
    # (For example, if we have 4 GPUs, the number of examples in each batch
    # must be divisible by 4.) We already ensured this for the batch_size, but
    # we have to additionally ensure that any "leftover" examples-- the remainder
    # examples (total examples % batch_size) that get called a batch for the very
    # last batch of an epoch-- do not raise an error when we try to split them
    # over the GPUs. This will likely be handled by Estimator during replication
    # in the future, but for now, we just drop the leftovers here.
    if multi_gpu:
        total_examples = num_epochs * examples_per_epoch
        dataset = dataset.take(batch_size * (total_examples // batch_size))

    # Parse the raw records into images and labels
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training),
                          num_parallel_calls=num_parallel_calls)

    dataset = dataset.batch(batch_size)

    # Operations between the final prefetch and the get_next call to the iterator
    # will happen synchronously during run time. We prefetch here again to
    # background all of the above processing work and keep it out of the
    # critical training path.
    dataset = dataset.prefetch(1)

    return dataset


def input_fn(is_training, data_dir, batch_size, num_epochs=1,
             num_parallel_calls=1, multi_gpu=False):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: The directory containing the input data.
      batch_size: The number of samples per batch.
      num_epochs: The number of epochs to repeat the dataset.
      num_parallel_calls: The number of records that are processed in parallel.
        This can be optimized per data set but for generally homogeneous data
        sets, should be approximately the number of available CPU cores.
      multi_gpu: Whether this is run multi-GPU. Note that this is only required
        currently to handle the batch leftovers, and can be removed
        when that is handled directly by Estimator.

    Returns:
      A dataset that can be used for iteration.
    """
    logging.info('searching %s for content', data_dir)
    filenames = get_filenames(is_training, data_dir)
    dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

    num_images = is_training and _NUM_IMAGES['train'] or _NUM_IMAGES['validation']

    ds = process_record_dataset(dataset, is_training, batch_size,
                                _NUM_IMAGES['train'],
                                parse_record, num_epochs, num_parallel_calls,
                                examples_per_epoch=num_images, multi_gpu=multi_gpu)
    logging.info("DATASET obtained")
    for k, v in ds.output_shapes.items():
        logging.info(">> {0} {1} {2}".format(k, ds.output_types[k], v))
    return ds


###############################################################################
# Running the model
###############################################################################
class Cifar10Model(resnet_model.Model):

    def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
                 version=resnet_model.DEFAULT_VERSION):
        """These are the parameters that work for CIFAR-10 data.

        Args:
          resnet_size: The number of convolutional layers needed in the model.
          data_format: Either 'channels_first' or 'channels_last', specifying which
            data format to use when setting up the model.
          num_classes: The number of output classes needed from the model. This
            enables users to extend the same model to their own datasets.
          version: Integer representing which version of the ResNet network to use.
            See README for details. Valid values: [1, 2]
        """
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(Cifar10Model, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            num_classes=num_classes,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            second_pool_size=8,
            second_pool_stride=1,
            block_sizes=[num_blocks] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            version=version,
            data_format=data_format)


def cifar10_model_fn(features, labels, mode, params):
    """Model function for CIFAR-10."""

    if not labels and isinstance(features, dict):
        labels = features['lbl']
    features = tf.reshape(features['img'] if isinstance(features, dict) else features,
                          [-1, _HEIGHT, _WIDTH, _NUM_CHANNELS])

    learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
        batch_size=params['batch_size'], batch_denom=128,
        num_images=_NUM_IMAGES['train'], boundary_epochs=[100, 150, 200],
        decay_rates=[1, 0.1, 0.01, 0.001])

    # We use a weight decay of 0.0002, which performs better
    # than the 0.0001 that was originally suggested.
    weight_decay = 2e-4

    # Empirical testing showed that including batch_normalization variables
    # in the calculation of regularized loss helped validation accuracy
    # for the CIFAR-10 dataset, perhaps because the regularization prevents
    # overfitting on the small data set. We therefore include all vars when
    # regularizing and computing loss during training.
    def loss_filter_fn(name):
        return True

    return resnet_run_loop.resnet_model_fn(features,
                                           labels,
                                           mode, Cifar10Model,
                                           resnet_size=params['resnet_size'],
                                           weight_decay=weight_decay,
                                           learning_rate_fn=learning_rate_fn,
                                           momentum=0.9,
                                           data_format=params['data_format'],
                                           version=params['version'],
                                           loss_filter_fn=loss_filter_fn,
                                           multi_gpu=params['multi_gpu'])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(argv=sys.argv)
