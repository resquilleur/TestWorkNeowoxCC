import abc
from abc import abstractmethod

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, LSTM, Activation, Flatten, Embedding
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D, GlobalAvgPool1D, MaxPooling1D, Dropout, \
    Conv1DTranspose


class Model(metaclass=abc.ABCMeta):

    @abstractmethod
    def make_architecture(self):
        pass


class NetLSTM(Model):

    def __init__(self):
        self.model = Sequential()

    def make_architecture(self, len, n_classes, k_emb=1, k_drop=7, k_filters=6):
        self.model.add(Embedding(len, 100 * k_emb))
        self.model.add(SpatialDropout1D(0.1 * k_drop))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(2 ** k_filters, dropout=0.1 * k_drop, recurrent_dropout=0.1 * k_drop))
        self.model.add(Dense(n_classes, activation='softmax'))


class NetConv(Model):

    def __init__(self):
        self.model = Sequential()

    def make_architecture(self):
        return self.model


class ModelFabric:

    def create_model(self, name):
        if name == 'lstm':
            return NetLSTM()

        if name == 'conv':
            return NetConv()
