import abc
from abc import abstractmethod

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dense, BatchNormalization, Conv1D, LSTM, Activation, Flatten, Embedding
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D, GlobalAvgPool1D, MaxPooling1D, Dropout, \
    Conv1DTranspose


class Model(metaclass=abc.ABCMeta):

    @staticmethod
    @abstractmethod
    def make_architecture(self):
        pass


class NetLSTM(Model):

    def make_architecture(self, params):
        len, n_classes, k_emb, k_drop, k_filters = params

        self.model = Sequential()

        self.model.add(Embedding(len, 100 * k_emb))
        self.model.add(SpatialDropout1D(0.1 * k_drop))
        self.model.add(BatchNormalization())
        self.model.add(LSTM(2 ** k_filters, dropout=0.1 * k_drop, recurrent_dropout=0.1 * k_drop))
        self.model.add(Dense(n_classes, activation='softmax'))
        return self.model


class NetConv(Model):

    def make_architecture(self, params):
        len, n_classes, k_emb, k_drop, k_kernel, k_filters, k_pool, k_dense = params

        self.model = Sequential()

        self.model.add(Embedding(len, 100 * k_emb))
        self.model.add(SpatialDropout1D(0.1 * k_drop))
        self.model.add(BatchNormalization())
        self.model.add(Conv1D(2**k_kernel, 1*k_filters, activation='relu', padding='same'))
        self.model.add(Conv1D(2**k_kernel, 1*k_filters, activation='relu', padding='same'))
        self.model.add(MaxPooling1D(1*k_pool))
        self.model.add(Dropout(0.1 * k_drop))
        self.model.add(BatchNormalization())
        self.model.add(GlobalAvgPool1D())
        self.model.add(Dense(2**k_dense, activation='relu'))
        self.model.add(Dropout(0.1 * k_drop))
        self.model.add(BatchNormalization())
        self.model.add(Dense(n_classes, activation='softmax'))

        return self.model


class ModelFabric:

    @staticmethod
    def create_model(name):
        if name == 'lstm':
            return NetLSTM()

        if name == 'conv':
            return NetConv()
