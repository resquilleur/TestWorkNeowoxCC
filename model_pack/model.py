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

    @abstractmethod
    def compile(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
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
        return self.model

    def compile(self, lr):
        self.model.compile(optimizer=Adam(learning_rate=lr, decay=1e-6), loss='categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def fit(self, x_train, y_train, x_test, y_test, callbacks, batch_size=16, epochs=30):
        history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks)
        return history

    def predict(self, x_test):
        pred = self.model.predict(x_test)
        return pred


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


class ModelFabric:

    def create_model(self, name):
        if name == 'lstm':
            return NetLSTM()

        if name == 'conv':
            return NetConv()
