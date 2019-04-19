import numpy as np


def get_model():
    return get_keras_model()


def initialize_keras_model():
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GaussianNoise
    from keras.models import Sequential

    drop_prob = 0.05

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(units=500, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=400, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=300, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=200, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=100, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=60, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=10, activation='softmax'))
    return model


def initialize_math_recognition_model():
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
    from keras.models import Sequential

    drop_prob = 0.1

    model = Sequential()
    model.add(Conv2D(input_shape=(45, 45, 1), filters=6, kernel_size=(5, 5),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=12, kernel_size=(5, 5),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=50, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Dense(units=14, activation='softmax'))
    return model


def get_keras_model():
    model = initialize_keras_model()

    model.load_weights('keras_model.h5')

    class Predictor:
        def predict(self, x):
            x = x.reshape(x.shape[0], 28, 28, 1)
            return model.predict(x)

    return Predictor()


def norm_generator(gen):
    for x_batch, y_batch in gen:
        yield x_batch / 255.0, y_batch


def train_mnist_model(learning_rate=0.001, epochs=10):
    from mnist import MNIST
    import keras
    from util import augmented_dataset_generator

    mndata = MNIST('./datasets/mnist')
    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    m_train = len(images)
    m_test = len(images_test)
    batch_size = 128

    images = images[:m_train]
    labels = labels[:m_train]
    images_test = images_test[:m_test]
    labels_test = labels_test[:m_test]

    gen = augmented_dataset_generator(images, labels, batch_size)
    Xtest = np.array(images_test, dtype=np.uint8).reshape((m_test, 28, 28, 1)) / 255.0

    Ytest = np.array(labels_test, dtype=np.uint8).reshape(m_test, 1)

    model = initialize_keras_model()

    optimizer = keras.optimizers.Adam(
        lr=learning_rate
    )
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    from math import ceil

    model.fit_generator(norm_generator(gen), epochs=epochs, steps_per_epoch=ceil(m_train / batch_size))

    gen = norm_generator(augmented_dataset_generator(images_test, labels_test, batch_size))

    loss_and_metrics = model.evaluate_generator(gen, steps=ceil(m_test / batch_size))
    print(loss_and_metrics)
    loss_and_metrics = model.evaluate(
        Xtest,
        keras.utils.to_categorical(Ytest, num_classes=10)
    )
    print(loss_and_metrics)
    model.save_weights('keras_model.h5')


def train_math_recognition_model():
    import os
    from dataset_utils import dataset_generator, dataset_size, load_dataset

    dir_path = os.path.join('datasets', 'digits_and_operators_csv')
    train_path = os.path.join(dir_path, 'train')
    dev_path = os.path.join(dir_path, 'dev')

    batch_size = 128
    m_train, _ = dataset_size(train_path)
    m_val, _ = dataset_size(dev_path)
    print('NUMBER OF TRAINING EXAMPLES', m_train)
    print('NUMBER OF DEV EXAMPLES', m_val)

    model = initialize_math_recognition_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x, labels = load_dataset(train_path)

    train_gen = dataset_generator(x, labels, mini_batch_size=batch_size)

    model.fit_generator(train_gen,
                        steps_per_epoch=int(m_train / batch_size), epochs=2)

    x_dev, labels_dev = load_dataset(dev_path)

    dev_gen = dataset_generator(x_dev, labels_dev, mini_batch_size=batch_size)
    res = model.evaluate_generator(dev_gen, steps=int(m_val / batch_size))
    print(res)
    model.save_weights('math_symbols_model.h5')


if __name__ == '__main__':
    import keras

    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('--lrate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--epochs', type=int, default=60,
                        help='number of iterations')

    args = parser.parse_args()

    #train_keras_model(args.lrate, args.epochs)
    train_math_recognition_model()
