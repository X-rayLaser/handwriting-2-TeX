def get_model():
    return get_math_symbols_model()


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


def get_math_symbols_model():
    model = initialize_math_recognition_model()
    model.load_weights('keras_model.h5')

    class Predictor:
        def predict(self, x):
            x = x.reshape(x.shape[0], 45, 45, 1)
            return model.predict(x)

    return Predictor()


def norm_generator(gen):
    for x_batch, y_batch in gen:
        yield x_batch / 255.0, y_batch


def train_math_recognition_model(epochs=2):
    import os
    from dataset_utils import dataset_generator, dataset_size, load_dataset

    dir_path = os.path.join('datasets', 'digits_and_operators_csv')
    train_path = os.path.join(dir_path, 'train')
    dev_path = os.path.join(dir_path, 'dev')
    test_path = os.path.join(dir_path, 'test')

    batch_size = 128
    m_train, _ = dataset_size(train_path)
    m_val, _ = dataset_size(dev_path)
    m_test, _ = dataset_size(test_path)

    print('NUMBER OF TRAINING EXAMPLES', m_train)
    print('NUMBER OF DEV EXAMPLES', m_val)
    print('NUMBER OF TEST EXAMPLES', m_test)

    model = initialize_math_recognition_model()

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x, labels = load_dataset(train_path)
    x_dev, labels_dev = load_dataset(dev_path)

    train_gen = dataset_generator(x, labels, mini_batch_size=batch_size)
    dev_gen = dataset_generator(x_dev, labels_dev, mini_batch_size=batch_size)

    model.fit_generator(train_gen,
                        steps_per_epoch=int(m_train / batch_size),
                        epochs=epochs,
                        validation_data=dev_gen,
                        validation_steps=int(m_val / batch_size))

    x_test, labels_test = load_dataset(test_path)
    test_gen = dataset_generator(x_test, labels_test, mini_batch_size=batch_size)

    res = model.evaluate_generator(test_gen, steps=int(m_test / batch_size))
    print(res)
    model.save_weights('keras_model.h5')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )

    parser.add_argument('--epochs', type=int, default=10,
                        help='number of iterations')

    args = parser.parse_args()

    train_math_recognition_model(epochs=args.epochs)
