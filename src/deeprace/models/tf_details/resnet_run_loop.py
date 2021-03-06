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
"""Contains utility and supporting functions for ResNet.

  This module contains ResNet code which does not directly build layers. This
includes dataset management, hyperparameter and optimizer code, and argument
parsing. Code for defining the ResNet layers can be found in resnet_model.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import logging
import tensorflow as tf
import numpy as np

from deeprace.utils.arg_parsers import parsers  # pylint: disable=g-bad-import-order
from deeprace.utils.logging import hooks_helper
from deeprace.utils.logging.hooks import timing_summary

from deeprace.models import resnet_model


def build_tensor_serving_input_receiver_fn(shape, dtype=tf.float32,
                                           batch_size=1):
    """Returns a input_receiver_fn that can be used during serving.

    This expects examples to come through as float tensors, and simply
    wraps them as TensorServingInputReceivers.

    Arguably, this should live in tf.estimator.export. Testing here first.

    Args:
      shape: list representing target size of a single example.
      dtype: the expected datatype for the input example
      batch_size: number of input tensors that will be passed for prediction

    Returns:
      A function that itself returns a TensorServingInputReceiver.
    """
    def serving_input_receiver_fn():
        # Prep a placeholder where the input example will be fed in
        features = tf.compat.v1.placeholder(
            dtype=dtype, shape=[batch_size] + shape, name='input_tensor')

        return tf.estimator.export.TensorServingInputReceiver(
            features=features, receiver_tensors=features)

    return serving_input_receiver_fn

################################################################################
# Functions for running training/eval/validation loops for the model.
################################################################################


def learning_rate_with_decay(
        batch_size, batch_denom, num_images, boundary_epochs, decay_rates):
    """Get a learning rate that decays step-wise as training progresses.

    Args:
      batch_size: the number of examples processed in each training batch.
      batch_denom: this value will be used to scale the base learning rate.
        `0.1 * batch size` is divided by this number, such that when
        batch_denom == batch_size, the initial learning rate will be 0.1.
      num_images: total number of images that will be used for training.
      boundary_epochs: list of ints representing the epochs at which we
        decay the learning rate.
      decay_rates: list of floats representing the decay rates to be used
        for scaling the learning rate. It should have one more element
        than `boundary_epochs`, and all elements should have the same type.

    Returns:
      Returns a function that takes a single argument - the number of batches
      trained so far (global_step)- and returns the learning rate to be used
      for training the next batch.
    """
    initial_learning_rate = 0.1 * batch_size / batch_denom
    batches_per_epoch = num_images / batch_size

    # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(global_step):
        global_step = tf.cast(global_step, tf.int32)
        return tf.train.piecewise_constant(global_step, boundaries, vals)

    return learning_rate_fn


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, version, loss_filter_fn=None, multi_gpu=False):
    """Shared functionality for different resnet model_fns.

    Initializes the ResnetModel representing the model layers
    and uses that model to build the necessary EstimatorSpecs for
    the `mode` in question. For training, this means building losses,
    the optimizer, and the train op that get passed into the EstimatorSpec.
    For evaluation and prediction, the EstimatorSpec is returned without
    a train op, but with the necessary parameters for the given mode.

    Args:
      features: tensor representing input images
      labels: tensor representing class labels for all input images
      mode: current estimator mode; should be one of
        `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
      model_class: a class representing a TensorFlow model that has a __call__
        function. We assume here that this is a subclass of ResnetModel.
      resnet_size: A single integer for the size of the ResNet model.
      weight_decay: weight decay loss rate used to regularize learned variables.
      learning_rate_fn: function that returns the current learning rate given
        the current global_step
      momentum: momentum term used for optimization
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      loss_filter_fn: function that takes a string variable name and returns
        True if the var should be included in loss calculation, and False
        otherwise. If None, batch_normalization variables will be excluded
        from the loss.
      multi_gpu: If True, wrap the optimizer in a TowerOptimizer suitable for
        data-parallel distribution across multiple GPUs.

    Returns:
      EstimatorSpec parameterized according to the input params and the
      current mode.
    """

    # Generate a summary node for the images
    tf.summary.image('images', features, max_outputs=6)

    model = model_class(resnet_size, data_format, version=version)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    if not loss_filter_fn:
        def loss_filter_fn(name):
            return 'batch_normalization' not in name

    # Add weight decay to the loss.
    loss = cross_entropy + weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()
         if loss_filter_fn(v.name)])

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loss, name='train_loss')
    tf.summary.scalar('train_loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=momentum)

        # If we are running multi-GPU, we need to wrap the optimizer.
        if multi_gpu:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(optimizer.minimize(loss, global_step), update_ops)
    else:
        train_op = None

    accuracy = tf.metrics.accuracy(
        tf.argmax(labels, axis=1), predictions['classes'])

    metrics = {'acc': accuracy}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_acc', accuracy[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


def validate_batch_size_for_multi_gpu(batch_size):
    """For multi-gpu, batch-size must be a multiple of the number of
    available GPUs.

    Note that this should eventually be handled by replicate_model_fn
    directly. Multi-GPU support is currently experimental, however,
    so doing the work here until that feature is in place.
    """
    from tensorflow.python.client import device_lib

    local_device_protos = device_lib.list_local_devices()
    num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
    if not num_gpus:
        raise ValueError('Multi-GPU mode was specified, but no GPUs '
                         'were found. To use CPU, run without --multi_gpu.')

    remainder = batch_size % num_gpus
    if remainder:
        err = ('When running with multiple GPUs, batch size '
               'must be a multiple of the number of available GPUs. '
               'Found {} GPUs with a batch size of {}; try --batch_size={} instead.'
               ).format(num_gpus, batch_size, batch_size - remainder)
        raise ValueError(err)


def resnet_main(flags, model_function, input_function, opts=None):
    # Using the Winograd non-fused algorithms provides a small performance boost.
    # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    epochs_per_eval = flags.train_epochs // flags.epochs_between_evals
    steps_per_epoch = int(opts["ntrain"]) // flags.batch_size

    logging.info('starting run on %i images (%i, %i)', int(opts['ntrain']), epochs_per_eval, steps_per_epoch)

    ngpus = 1
    if opts:
        ngpus = int(opts["n_gpus"])

    if flags.batch_size != int(opts["batch_size"]):
        logging.warning("batch sizes differ in model %i %s", flags.batch_size, opts["batch_size"])

    if ngpus > 1:
        steps_per_epoch -= 1
        validate_batch_size_for_multi_gpu(flags.batch_size)
        # There are two steps required if using multi-GPU: (1) wrap the model_fn,
        # and (2) wrap the optimizer. The first happens here, and (2) happens
        # in the model_fn itself when the optimizer is defined.
        model_function = tf.contrib.estimator.replicate_model_fn(
            model_function,
            loss_reduction=tf.losses.Reduction.MEAN,
            devices=None)

    # Create session config based on values of inter_op_parallelism_threads and
    # intra_op_parallelism_threads. Note that we default to having
    # allow_soft_placement = True, which is required for multi-GPU and not
    # harmful for other modes.
    session_config = tf.ConfigProto(
        inter_op_parallelism_threads=flags.inter_op_parallelism_threads,
        intra_op_parallelism_threads=flags.intra_op_parallelism_threads,
        allow_soft_placement=True)

    # Set up a RunConfig to save checkpoint and set session config.
    if opts and opts["checkpoint_epochs"]:
        run_config = tf.estimator.RunConfig().replace(save_checkpoints_steps=epochs_per_eval,
                                                      session_config=session_config)
    else:
        run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9,
                                                      save_checkpoints_steps=None,
                                                      # keep_checkpoint_every_n_hours=int(1e4),
                                                      session_config=session_config)

    classifier = tf.estimator.Estimator(
        model_fn=model_function,
        model_dir=flags.model_dir,
        config=run_config,
        params={
            'resnet_size': flags.resnet_size,
            'data_format': flags.data_format,
            'batch_size': flags.batch_size,
            'multi_gpu': flags.multi_gpu,
            'version': flags.version,
        })

    flags.hooks.append("TimePerEpochHook")
    flags.hooks.append("CaptureTensorsHook")

    def input_fn_train():
        return input_function(True,
                              flags.data_dir,
                              flags.batch_size,
                              flags.epochs_between_evals,
                              flags.num_parallel_calls,
                              flags.multi_gpu)

    # Evaluate the model and print results
    def input_fn_eval():
        return input_function(False, flags.data_dir, flags.batch_size,
                              1, flags.num_parallel_calls, flags.multi_gpu)

    global_times = timing_summary()
    history = {}

    #######################################################################################
    # TRAINING LOOP
    #train_hooks = None
    for _ in range(epochs_per_eval):
        train_hooks = hooks_helper.get_train_hook_dict(flags.hooks,
                                                       batch_size=flags.batch_size,
                                                       every_n_steps=steps_per_epoch,
                                                       tensors=['train_accuracy', 'train_loss'])

        logging.info('Starting a training cycle. %s', train_hooks.keys())

        classifier.train(input_fn=input_fn_train,
                         hooks=train_hooks.values(),
                         max_steps=flags.max_train_steps)

        logging.info('Starting to evaluate.')

        # flags.max_train_steps is generally associated with testing and profiling.
        # As a result it is frequently called with synthetic data, which will
        # iterate forever. Passing steps=flags.max_train_steps allows the eval
        # (which is generally unimportant in those circumstances) to terminate.
        # Note that eval will run for max_train_steps each loop, regardless of the
        # global_step count.
        validation_results = classifier.evaluate(input_fn=input_fn_eval,
                                                 steps=flags.max_train_steps)

        for k in validation_results.keys():
            if "global_step" in k:
                continue
            value = validation_results[k]

            if k in history.keys():
                history[k].append(value)
            else:
                history[k] = [value]

        for k in train_hooks["CaptureTensorsHook"].captured.keys():
            if k in history.keys():
                history[k].extend(train_hooks["CaptureTensorsHook"].captured[k])
            else:
                history[k] = train_hooks["CaptureTensorsHook"].captured[k]

        epoch_times = train_hooks["TimePerEpochHook"].summary()
        global_times.add(epoch_times)

    #######################################################################################

    # don't ask about the following I am happy I got this far
    history["val_loss"] = history.pop("loss")
    history["val_acc"] = history.pop("acc")
    history["loss"] = history.pop("train_loss")
    history["acc"] = history.pop("train_accuracy")

    # export_dtype = flags_core.get_tf_dtype(flags_obj)
    # if flags_obj.image_bytes_as_serving_input:
    #   input_receiver_fn = functools.partial(
    #     image_bytes_serving_input_fn, shape, dtype=export_dtype)
    # else:
    input_receiver_fn = build_tensor_serving_input_receiver_fn(
        [32, 32, 3], batch_size=flags.batch_size, dtype=tf.float32)

    servable_model_path = classifier.export_savedmodel(flags.model_dir, input_receiver_fn, as_text=True, strip_default_attrs=True)

    logging.info("stored model to {0}".format(servable_model_path))
    logging.info(
        "Completed %i epochs (acc %i, val_acc %i)", len(
            global_times.epoch_durations), len(
            history["acc"]), len(
                history["val_acc"]))
    return history, global_times
