import keras
from keras.callbacks import Callback
import datetime as dt

class stopwatch(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_begin = dt.datetime.now()
        self.epoch_durations = []
        self.epoch_start = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start.append(dt.datetime.now())

    def on_epoch_end(self, batch, logs={}):
        self.epoch_durations.append((dt.datetime.now() - self.epoch_start[-1]).total_seconds())

    def on_train_end(self, logs={}):
        self.train_end = dt.datetime.now()
