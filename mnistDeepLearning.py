import sys
import numpy as np
import numpy.random as rd
rd.seed(7)
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, SimpleRNN, GRU
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def encode(data):
    encoded = to_categorical(data)
    return encoded

class MnistClassifier(object):
    def __init__(self, model_type, n_layers = 1, n_epochs = 10):
        # Classifier
        self.model_type=model_type # timesteps to unroll
        self.time_steps=28 # timesteps to unroll
        self.n_units=128 # hidden LSTM units
        self.n_inputs=28 # rows of 28 pixels (an mnist img is 28x28)
        self.n_classes=10 # mnist classes/labels (0-9)
        self.batch_size=128 # Size of each batch
        self.n_epochs=n_epochs
        self.n_layers = n_layers
        # Internal
        self._data_loaded = False
        self._trained = False

    def __create_model(self):
        K.clear_session()
        self.model = Sequential()
        
        if self.model_type == 'LSTM':
            if self.n_layers == 1:
                self.model.add(LSTM(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
            else:
                self.model.add(LSTM(self.n_units,
                                    input_shape=(self.time_steps, self.n_inputs),
                                    return_sequences = True))
            for i in range(2, self.n_layers+1):
                if i < self.n_layers:
                    self.model.add(LSTM(self.n_units,
                                        return_sequences = True))
                else:
                    self.model.add(LSTM(self.n_units,
                                        return_sequences = False))
        elif self.model_type == 'RNN':
            if self.n_layers == 1:
                self.model.add(SimpleRNN(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
            else:
                self.model.add(SimpleRNN(self.n_units,
                                    input_shape=(self.time_steps, self.n_inputs),
                                    return_sequences = True))
            for i in range(2, self.n_layers+1):
                if i < self.n_layers:
                    self.model.add(SimpleRNN(self.n_units,
                                        return_sequences = True))
                else:
                    self.model.add(SimpleRNN(self.n_units,
                                        return_sequences = False))
        elif self.model_type == 'GRU':
            if self.n_layers == 1:
                self.model.add(GRU(self.n_units, input_shape=(self.time_steps, self.n_inputs)))
            else:
                self.model.add(GRU(self.n_units,
                                    input_shape=(self.time_steps, self.n_inputs),
                                    return_sequences = True))
            for i in range(2, self.n_layers+1):
                if i < self.n_layers:
                    self.model.add(GRU(self.n_units,
                                        return_sequences = True))
                else:
                    self.model.add(GRU(self.n_units,
                                        return_sequences = False))
                             
        elif self.model_type == 'Dense':
            for i in range(1,self.n_layers+1):
                if i == 1:
                    self.model.add(Dense(self.n_units, 
                                         input_shape=(self.n_inputs**2,),
                                         activation = 'sigmoid'))
                else:
                    self.model.add(Dense(self.n_units, 
                                         activation = 'sigmoid'))
        
        self.model.add(Dense(self.n_classes, activation='softmax'))
        print(self.model.summary())

        self.model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

    def __load_data(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
        self.y_train = encode(self.y_train)
        self.y_test = encode(self.y_test)


        self._data_loaded = True

    def train(self, save_model=False):
        self.__create_model()
        if self._data_loaded == False:
            self.__load_data()
        if self.model_type == 'Dense':
            self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1]*self.x_train.shape[2])
            self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1]*self.x_test.shape[2])
            print(self.x_test.shape)
            
        self.model.fit(self.x_train, 
                       self.y_train,
                       batch_size=self.batch_size, 
                       epochs=self.n_epochs,
                       validation_data = [self.x_test, self.y_test],
                       shuffle=False)

        self._trained = True
        
        if save_model:
            self.model.save("%s-model_mnist_fashion.h5" % self.model_type)

    def evaluate(self, model=None):
        if self._trained == False and model == None:
            errmsg = "[!] Error: classifier wasn't trained or classifier path is not precised."
            print(errmsg, file=sys.stderr)
            sys.exit(0)

        if self._data_loaded == False:
            self.__load_data()

        model = load_model(model) if model else self.model
        test_loss = model.evaluate(self.x_test, self.y_test)
        print(test_loss)

dense_classifier = MnistClassifier('Dense')
dense_classifier.train(save_model = True)
dense_classifier.evaluate()

lstm_classifier = MnistClassifier('LSTM')
lstm_classifier.train(save_model = True)
lstm_classifier.evaluate()

rnn_classifier = MnistClassifier('RNN')
rnn_classifier.train(save_model = True)
rnn_classifier.evaluate()

gru_classifier = MnistClassifier('GRU')
gru_classifier.train(save_model = True)
gru_classifier.evaluate()

dense_history = dense_classifier.model.history.history
rnn_history = rnn_classifier.model.history.history
lstm_history = lstm_classifier.model.history.history
gru_history = gru_classifier.model.history.history

histories = {'dense': dense_history,
             'rnn': rnn_history, 
             'lstm': lstm_history,
             'gru': gru_history}


n_epochs = len(dense_history['val_loss'])
plt.figure()
for h in histories.keys():
    plt.plot(range(n_epochs), histories[h]['val_loss'], label = h)
plt.savefig('mnistDeepLearning1.png')
plt.legend()
plt.show()

plt.figure()
for h in histories.keys():
    plt.plot(range(n_epochs), histories[h]['val_acc'], label = h)
plt.savefig('mnistDeepLearning2.png')
plt.legend()
plt.show()