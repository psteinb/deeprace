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

"""Hook that counts examples per second every N steps or seconds."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tensorflow as tf
import datetime as dt

from tensorflow.python.framework import ops
import six

class timing_summary:

  def __init__(self):

    self.train_begin = None
    self.epoch_durations = None
    self.epoch_start = None
    self.train_end = None

  def add(self,other):
    self.train_begin = other.train_begin
    self.train_end = other.train_end

    if not self.epoch_durations:
      self.epoch_durations = other.epoch_durations
      self.epoch_start = other.epoch_start
    else:
      self.epoch_durations.extend(other.epoch_durations)
      self.epoch_start.extend(other.epoch_start)

class TimePerEpochHook(tf.train.SessionRunHook):
  def __init__(self,
               every_n_steps,
               warm_steps=-1):

    self.every_n_steps = every_n_steps
    logging.info("TimePerEpochHook triggering every %i steps",every_n_steps)
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=self.every_n_steps #call timer after each epoch
    )

    self._epoch_train_time = 0
    self._total_steps = 0
    self._warm_steps = warm_steps

    self._train_begin = None
    self._train_end = None
    self.epoch_durations = []
    self._epoch_start = []
    self._step = -1


  def summary(self):

    value = timing_summary()
    value.train_begin = self._train_begin
    value.train_end = dt.datetime.now()
    value.epoch_durations = self.epoch_durations
    value.epoch_start = self._epoch_start

    return value

  def begin(self):
    """Called once before using the session to check global step."""
    self._train_begin = dt.datetime.now()
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run().

    Args:
      run_context: A SessionRunContext object.

    Returns:
      A SessionRunArgs object or None if never triggered.
    """
    self._step += 1
    # gs = tf.train.global_step(run_context, self._global_step_tensor)
    if self._step % self.every_n_steps == 0:
      self._epoch_start.append((dt.datetime.now() - self._train_begin).total_seconds())
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Called after each call to run().

    Args:
      run_context: A SessionRunContext object.
      run_values: A SessionRunValues object.
    """
    global_step = run_values.results
    sess = run_context.session

    #if self._timer.should_trigger_for_step(global_step) and global_step > self._warm_steps:
    if self._step % self.every_n_steps == 0:
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        self._epoch_train_time += elapsed_time
        self._total_steps += elapsed_steps

        self.epoch_durations.append(self._epoch_train_time)
        tf.logging.info('Epoch [%g steps]: %g (%s)', self._total_steps,self._epoch_train_time,str(self.epoch_durations))

        self._epoch_train_time = 0
      else:
        logging.warning("step %i, elapsed_time is None!", global_step)



class ExamplesPerSecondHook(tf.train.SessionRunHook):
  """Hook to print out examples per second.

  Total time is tracked and then divided by the total number of steps
  to get the average step time and then batch_size is used to determine
  the running average of examples per second. The examples per second for the
  most recent interval is also logged.
  """

  def __init__(self,
               batch_size,
               every_n_steps=None,
               every_n_secs=None,
               warm_steps=0):
    """Initializer for ExamplesPerSecondHook.

    Args:
      batch_size: Total batch size across all workers used to calculate
        examples/second from global time.
      every_n_steps: Log stats every n steps.
      every_n_secs: Log stats every n seconds. Exactly one of the
        `every_n_steps` or `every_n_secs` should be set.
      warm_steps: The number of steps to be skipped before logging and running
        average calculation. warm_steps steps refers to global steps across all
        workers, not on each worker

    Raises:
      ValueError: if neither `every_n_steps` or `every_n_secs` is set, or
      both are set.
    """

    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError('exactly one of every_n_steps'
                       ' and every_n_secs should be provided.')

    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps, every_secs=every_n_secs)

    self._step_train_time = 0
    self._total_steps = 0
    self._batch_size = batch_size
    self._warm_steps = warm_steps

  def begin(self):
    """Called once before using the session to check global step."""
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          'Global step should be created to use StepCounterHook.')

  def before_run(self, run_context):  # pylint: disable=unused-argument
    """Called before each call to run().

    Args:
      run_context: A SessionRunContext object.

    Returns:
      A SessionRunArgs object or None if never triggered.
    """
    return tf.train.SessionRunArgs(self._global_step_tensor)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    """Called after each call to run().

    Args:
      run_context: A SessionRunContext object.
      run_values: A SessionRunValues object.
    """
    global_step = run_values.results

    if self._timer.should_trigger_for_step(
        global_step) and global_step > self._warm_steps:
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        self._step_train_time += elapsed_time
        self._total_steps += elapsed_steps

        # average examples per second is based on the total (accumulative)
        # training steps and training time so far
        average_examples_per_sec = self._batch_size * (
            self._total_steps / self._step_train_time)
        # current examples per second is based on the elapsed training steps
        # and training time per batch
        current_examples_per_sec = self._batch_size * (
            elapsed_steps / elapsed_time)
        # Current examples/sec followed by average examples/sec
        tf.logging.info('Batch [%g]:  current exp/sec = %g, average exp/sec = '
                        '%g', self._total_steps, current_examples_per_sec,
                        average_examples_per_sec)


class CaptureTensorsHook(tf.train.SessionRunHook):
  """Prints the given tensors every N local steps, every N seconds, or at end.
  The tensors will be printed to the log, with `INFO` severity. If you are not
  seeing the logs, you might want to add the following line after your imports:
  ```python
    tf.logging.set_verbosity(tf.logging.INFO)
  ```
  Note that if `at_end` is True, `tensors` should not include any tensor
  whose evaluation produces a side effect such as consuming additional inputs.
  """

  def __init__(self, tensors, every_n_steps=None,at_end=False):
    """Initializes a `CaptureTensorHook`.
    Args:
      tensors: `dict` that maps string-valued tags to tensors/tensor names,
          or `iterable` of tensors/tensor names.
      every_n_iter: `int`, print the values of `tensors` once every N local
          steps taken on the current worker.
      at_end: `bool` specifying whether to print the values of `tensors` at the
          end of the run.

    Raises:
      ValueError: if `every_n_iter` is non-positive.
    """
    only_log_at_end = (at_end and (every_n_steps is None))
    if (not only_log_at_end and every_n_steps is None):
      raise ValueError(
          "either at_end and/or exactly one of every_n_steps and every_n_secs "
          "must be provided.")
    if every_n_steps is not None and every_n_steps <= 0:
      raise ValueError("invalid every_n_steps=%s." % every_n_steps)

    self.captured = {}

    if not isinstance(tensors, dict):
      self._tag_order = tensors
      tensors = {item: item for item in tensors}

    else:
      self._tag_order = tensors.keys()
    self._tensors = tensors
    for k in self._tensors.keys():
      self.captured[k] = []

    self._timer = (
        NeverTriggerTimer() if only_log_at_end else
        tf.train.SecondOrStepTimer(every_steps=every_n_steps))
    self._log_at_end = at_end
    self.period = every_n_steps

  def begin(self):
    self._timer.reset()
    self._iter_count = 0
    # Convert names to tensors if given
    self._current_tensors = {tag: _as_graph_element(tensor)
                             for (tag, tensor) in self._tensors.items()}

  def before_run(self, run_context):  # pylint: disable=unused-argument
     self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
     if self._should_trigger:
       return tf.train.SessionRunArgs(self._current_tensors)
     else:
       return None

  def _log_tensors(self, tensor_values):

    logging.debug("%i/%i %s",self._iter_count,self.period,tensor_values)

    #TODO: something is wrong with the counting of the steps per epoch
    #TODO: added this fix after manual validation of logs with default tf output
    if self._iter_count == self.period:
      return

    for (k,v) in tensor_values.items():
      self.captured[k].append(v)
    # original = np.get_printoptions()
    # np.set_printoptions(suppress=True)
    # elapsed_secs, _ = self._timer.update_last_triggered_step(self._iter_count)
    # if self._formatter:
    #   logging.info(self._formatter(tensor_values))
    # else:
    #   stats = []
    #   for tag in self._tag_order:
    #     stats.append("%s = %s" % (tag, tensor_values[tag]))
    #   if elapsed_secs is not None:
    #     logging.info("%s (%.3f sec)", ", ".join(stats), elapsed_secs)
    #   else:
    #     logging.info("%s", ", ".join(stats))
    # np.set_printoptions(**original)

    

  def after_run(self, run_context, run_values):
    _ = run_context

    if self._timer.should_trigger_for_step(self._iter_count):
      self._timer.update_last_triggered_step(self._iter_count)
      self._log_tensors(run_values.results)

    self._iter_count += 1

  def end(self, session):
    if self._log_at_end:
      values = session.run(self._current_tensors)
      self._log_tensors(values)

def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element
