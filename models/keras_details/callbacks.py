import keras
from keras.callbacks import Callback
import datetime as dt

class stopwatch(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_begin = dt.datetime.now()
        self.train_end = None
        self.epoch_durations = []
        self.epoch_start = []
        self.epoch_start_dt = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_start_dt.append(dt.datetime.now())
        self.epoch_start.append((dt.datetime.now() - self.train_begin).total_seconds())

    def on_epoch_end(self, batch, logs={}):
        self.epoch_durations.append((dt.datetime.now() - self.epoch_start_dt[-1]).total_seconds())

    def on_train_end(self, logs={}):
        self.train_end = dt.datetime.now()
