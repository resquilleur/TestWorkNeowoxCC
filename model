from __future__ import annotations
import abc
from abc import abstractmethod

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv1D, LSTM, Activation, Flatten, Embedding
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D, GlobalAvgPool1D, MaxPooling1D, Dropout, \
    Conv1DTranspose


class Model(metaclass=abc.ABCMeta):

    @abstractmethod
    def make_architecture(self):
        pass

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def get_result(self):
        pass


class NetLSTM(Model):

    def __init__(self):
        self.model = Sequential()
        self.history = []
        self.pred = []
        self.result = []

    def make_architecture(self):
        return self.model

    def compile(self):
        return self.model

    def fit(self):
        return self.history

    def predict(self):
        return self.pred

    def get_result(self):
        return self.result


class NetConv(Model):

    def __init__(self):
        self.model = Sequential()
        self.history = []
        self.pred = []
        self.result = []

    def make_architecture(self):
        return self.model

    def compile(self):
        return self.model

    def fit(self):
        return self.history

    def predict(self):
        return self.pred

    def get_result(self):
        return self.result


class ModelFabric:

    def CreateModel(self, name):
        if name == 'lstm':
            return NetLSTM()

        if name == 'conv':
            return NetConv()
