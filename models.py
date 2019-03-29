import os
import json
import numpy as np


class DigitsNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.initialize(input_size, hidden_size, output_size)

    def initialize(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))

        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def train(self, X, Y, learning_rate, epochs):
        for i in range(epochs):
            A2, cache = self.forward_propagation(X)
            gradients = self.back_propagation(X, A2, Y, cache)
            dW1 = gradients['dW1']
            db1 = gradients['db1']
            dW2 = gradients['dW2']
            db2 = gradients['db2']
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            print('Cost {}'.format(self.compute_cost(Y, A2)))

    def back_propagation(self, X, A2, Y, cache):
        m = Y.shape[1]

        Z1 = cache['Z1']
        A1 = cache['A1']

        dZ2 = A2 - Y
        dW2 = 1.0 / m * np.dot(dZ2, A1.T)
        db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.W2.T, dZ2) * self.gprime(Z1)
        dW1 = 1.0 / m * np.dot(dZ1, X.T)
        db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

        return {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1
        }

    def forward_propagation(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        cache = {
            'Z1': Z1,
            'A1': A1
        }
        return A2, cache

    def compute_cost(self, Y, Y_hat):
        m = Y.shape[1]
        return - np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / m

    def gprime(self, Z):
        g = np.tanh(Z)
        return 1 - np.power(g, 2)

    def sigmoid(self, Z):
        return 1.0 / (1 + np.exp(-Z))

    def softmax(self, Z):
        a = np.exp(Z)
        return a / np.sum(a, axis=0, keepdims=True)

    def predict_all(self, X):
        A2, cache = self.forward_propagation(X)
        return np.argmax(A2, axis=0)

    def predict(self, x):
        x = x.reshape(28**2, 1)
        A2, cache = self.forward_propagation(x)
        p = np.max(A2)
        return np.argmax(A2), p

    def save(self, fname):
        with open(fname, 'w') as f:
            d = {
                'W2': self.W2.tolist(),
                'b2': self.b2.tolist(),
                'W1': self.W1.tolist(),
                'b1': self.b1.tolist()
            }

            f.write(json.dumps(d))

    def restore(self, d):
        self.W2 = d['W2']
        self.b2 = d['b2']
        self.W1 = d['W1']
        self.b1 = d['b1']


def get_model():
    return get_keras_model()
    if not os.path.isfile('mnist_model.json'):
        raise Exception('Model file not found')

    with open('mnist_model.json') as f:
        s = f.read()
    d = json.loads(s)

    net = DigitsNet(28*28, 30, 10)
    net.restore(d)
    return net


def initialize_keras_model():
    from keras.layers import Dense, Dropout
    from keras.models import Sequential

    drop_prob = 0.

    model = Sequential()
    model.add(Dense(units=500, activation='relu', input_dim=28**2))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=400, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=300, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=200, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=60, activation='relu'))
    model.add(Dropout(drop_prob))

    model.add(Dense(units=10, activation='softmax'))
    return model


def get_keras_model():
    model = initialize_keras_model()

    model.load_weights('keras_model.h5')

    class Predictor:
        def predict(self, x):
            x = x.reshape(28 ** 2, 1).T

            print('SHAPE', x.shape)
            a = model.predict(x)[0]
            p = np.max(a)
            return np.argmax(a), p

    return Predictor()


def estimate_accuracy(net, X, labels):
    Y_hat = net.predict_all(X)

    labels = np.array(labels)
    diff = np.round(np.abs(Y_hat - labels))
    misclass = np.sum(diff > 0)

    mtest = X.shape[1]

    print(
        'Accurcay {} %'.format((mtest - misclass) / float(mtest) * 100)
    )


def extract_dataset_mean(Xtrain):
    with open('mnist_info.json', 'w') as f:
        nx, m = Xtrain.shape
        mu = np.mean(Xtrain, axis=1, keepdims=True)
        d = {
            'mu': mu.tolist()
        }
        f.write(json.dumps(d))


def normalize(X):
    if not os.path.isfile('mnist_info.json'):
        raise Exception('File {} does not exist'.format('mnist_info.json'))

    with open('mnist_info.json', 'r') as f:
        s = f.read()
        d = json.loads(s)
        mu = np.array(d['mu'])

    return (X - mu) / 255.0


def train_keras_model(learning_rate=0.001, epochs=10):
    from mnist import MNIST
    import keras

    mndata = MNIST('./datasets/mnist')
    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    m_train = len(images)
    m_test = len(images_test)
    #m_train = 10000
    #m_test = 10000

    images = images[:m_train]
    labels = labels[:m_train]
    images_test = images_test[:m_test]
    labels_test = labels_test[:m_test]

    Xtrain = np.array(images, dtype=np.uint8)
    Xtest = np.array(images_test, dtype=np.uint8)

    Xtrain_norm = Xtrain / 255.0
    Xtest_norm = Xtest / 255.0

    Ytrain = np.array(labels, dtype=np.uint8).reshape(m_train, 1)
    Ytest = np.array(labels_test, dtype=np.uint8).reshape(m_test, 1)

    Ytrain = keras.utils.to_categorical(Ytrain, num_classes=10)
    Ytest = keras.utils.to_categorical(Ytest, num_classes=10)

    model = initialize_keras_model()

    optimizer = keras.optimizers.SGD(
        lr=learning_rate, momentum=0.9, decay=0.0, nesterov=False
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    model.fit(Xtrain_norm, Ytrain, epochs=epochs, batch_size=32)

    loss_and_metrics = model.evaluate(Xtest_norm, Ytest, batch_size=128)
    print(loss_and_metrics)
    model.save_weights('keras_model.h5')


def train_model(learning_rate=0.001, epochs=10):
    from mnist import MNIST
    mndata = MNIST('./datasets/mnist')
    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    Xtrain = np.array(images, dtype=np.uint8).T
    Xtest = np.array(images_test, dtype=np.uint8).T

    extract_dataset_mean(Xtrain)

    Xtrain_norm = normalize(Xtrain)
    Xtest_norm = normalize(Xtest)

    mtrain = Xtrain.shape[1]
    mtest = Xtest.shape[1]

    Ytrain = np.zeros((10, mtrain), dtype=np.float)
    Ytest = np.zeros((10, mtest), dtype=np.float)

    for i in range(mtrain):
        label = labels[i]
        Ytrain[label, i] = 1.0

    for i in range(mtest):
        label = labels_test[i]
        Ytest[label, i] = 1.0

    net = DigitsNet(28*28, 100, 10)
    net.train(Xtrain_norm, Ytrain, learning_rate=learning_rate, epochs=epochs)

    estimate_accuracy(net, Xtrain_norm, labels)
    estimate_accuracy(net, Xtest_norm, labels_test)

    net.save('mnist_model.json')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('--lrate', type=float, default=0.003,
                        help='learning rate')

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of iterations')

    args = parser.parse_args()

    #train_model(args.lrate, args.epochs)
    train_keras_model(args.lrate, args.epochs)
