import numpy as np


def get_model():
    return get_keras_model()


def initialize_keras_model():
    from keras.layers import Dense, Dropout, Flatten, BatchNormalization
    from keras.models import Sequential

    drop_prob = 0.1

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


def get_keras_model():
    model = initialize_keras_model()

    model.load_weights('keras_model.h5')

    class Predictor:
        def predict(self, x):
            x = x.reshape(1, 28, 28, 1)

            print('SHAPE', x.shape)
            a = model.predict(x)[0]
            p = np.max(a)
            return np.argmax(a), p

    return Predictor()


def norm_generator(gen):
    for x_batch, y_batch in gen:
        yield x_batch / 255.0, y_batch


def train_keras_model(learning_rate=0.001, epochs=10):
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('--lrate', type=float, default=0.001,
                        help='learning rate')

    parser.add_argument('--epochs', type=int, default=30,
                        help='number of iterations')

    args = parser.parse_args()

    train_keras_model(args.lrate, args.epochs)
