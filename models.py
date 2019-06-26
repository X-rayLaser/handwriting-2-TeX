from keras.layers import Input, InputLayer, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from keras import Model
import argparse
from keras.datasets import mnist
import os


class ModelBuilder:
    def __init__(self, input_shape, initial_num_filters=6):
        self._filters = initial_num_filters
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=input_shape))
        h, w, d = input_shape
        self._input_width = w
        self._input_height = h

    def add_conv_layer(self, last_one=False, kernel_size=(3, 3)):
        kwargs = dict(filters=self._filters, kernel_size=kernel_size,
                      kernel_initializer='he_normal', activation='relu')
        if last_one:
            kwargs['name'] = 'last_feature_extraction_layer'

        layer = Conv2D(**kwargs)

        self._model.add(layer)
        self._filters *= 2
        self._input_height -= (kernel_size[0] - 1)
        self._input_width -= (kernel_size[1] - 1)

        return self

    def add_pooling_layer(self):
        self._model.add(MaxPool2D(pool_size=(2, 2)))
        self._input_width = self._input_width // 2
        self._input_height = self._input_height // 2

        return self

    def add_batch_norm_layer(self):
        self._model.add(BatchNormalization())
        return self

    def add_dropout_layer(self, drop_prob=0.5):
        self._model.add(Dropout(drop_prob))
        return self

    def add_fully_connected_layer(self):
        self._model.add(
            Conv2D(filters=self._filters,
                   kernel_size=(self._input_height, self._input_width),
                   kernel_initializer='he_normal', activation='relu')
        )

        self._input_height = 1
        self._input_width = 1
        return self

    def add_output_layer(self, num_classes):
        self._model.add(Conv2D(filters=num_classes, kernel_size=(1, 1),
                               kernel_initializer='he_normal',
                               activation='softmax')
                        )
        return self

    def add_binary_classification_layer(self):
        self._model.add(Conv2D(filters=1, kernel_size=(1, 1),
                               kernel_initializer='he_normal',
                               activation='sigmoid')
                        )
        return self

    def load_weights(self, path):
        self._model.load_weights(path)

    def get_complete_model(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp
        for layer in self._model.layers:
            x = layer(x)

        return Model(input=inp, output=x)

    def index_of_last_extraction_layer(self):
        for i in range(len(self._model.layers)):
            layer = self._model.layers[i]
            if layer.name == 'last_feature_extraction_layer':
                return i

    def get_feature_extractor(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp

        for i in range(self.index_of_last_extraction_layer() + 1):
            layer = self._model.layers[i]
            x = layer(x)

        out = x
        return Model(input=inp, output=out)

    def get_classifier(self, input_shape):
        inp = Input(shape=input_shape)
        x = inp

        pooling_index = self.index_of_last_extraction_layer() + 1

        for i in range(pooling_index, len(self._model.layers)):
            layer = self._model.layers[i]
            x = layer(x)

        out = x
        return Model(input=inp, output=out)


def build_classification_model(input_shape, num_classes):

    builder = ModelBuilder(input_shape=input_shape, initial_num_filters=6)
    builder.add_conv_layer(kernel_size=(5, 5)).add_batch_norm_layer()
    builder.add_conv_layer(kernel_size=(5, 5))
    builder.add_pooling_layer().add_batch_norm_layer()

    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer()
    builder.add_pooling_layer().add_batch_norm_layer()

    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer(0.1)
    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer(0.1)

    builder.add_output_layer(num_classes=num_classes)
    return builder


def calculate_num_steps(num_examples, batch_size):
    n = int(num_examples / batch_size)
    if n == 0:
        n = 1

    return n


def train_model(model, train_gen, validation_gen, m_train, m_val, mini_batch_size=32,
                loss='categorical_crossentropy', metrics=None,
                save_path='trained_model.h5', epochs=6):
    if metrics is None:
        metrics = ['accuracy']

    model_to_train = model
    model_to_train.compile(optimizer='adam', loss=loss, metrics=metrics)

    model_to_train.fit_generator(
        generator=train_gen,
        steps_per_epoch=calculate_num_steps(m_train, mini_batch_size),
        epochs=epochs,
        validation_data=validation_gen,
        validation_steps=calculate_num_steps(m_val, mini_batch_size)
    )
    model_to_train.save_weights(save_path)


def combined_generator(mnist_data, dir_path, mini_batch_size=128):
    from dataset_utils import load_dataset, dataset_generator
    from skimage.transform import resize
    from skimage import img_as_ubyte
    x, y = load_dataset(dir_path)

    x_mnist, y_mnist = mnist_data

    size = (45, 45)

    import numpy as np
    x_mnist_scaled = np.zeros((len(x_mnist), 45 * 45), dtype=np.uint8)
    for i in range(len(x_mnist)):
        resized = resize(x_mnist[i], size, anti_aliasing=True)
        resized = img_as_ubyte(resized)

        x_mnist_scaled[i] = resized.reshape(45 * 45)

    x = np.vstack((x, x_mnist_scaled))
    y = np.hstack((y, y_mnist))

    m = len(y)

    gen = dataset_generator(x, y, mini_batch_size=mini_batch_size)

    def wrapped_gen():
        for x_batch, y_batch in gen:
            yield x_batch, y_batch.reshape((-1, 1, 1, 14))

    return wrapped_gen(), m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=''
    )

    parser.add_argument('--epochs', type=int, default=9,
                        help='number of iterations')

    args = parser.parse_args()

    (x_train_dev, y_train_dev), mnist_test = mnist.load_data()

    m_train = 55000
    mini_batch_size = 64

    x_train = x_train_dev[:m_train]
    y_train = y_train_dev[:m_train]

    x_dev = x_train_dev[m_train:]
    y_dev = y_train_dev[m_train:]

    train_path = os.path.join('datasets', 'digits_and_operators_csv', 'train')
    dev_path = os.path.join('datasets', 'digits_and_operators_csv', 'dev')

    train_gen, m_train = combined_generator((x_train, y_train), train_path,
                                            mini_batch_size)
    val_gen, m_dev = combined_generator((x_dev, y_dev), dev_path,
                                        mini_batch_size)

    builder = build_classification_model(input_shape=(45, 45, 1), num_classes=14)
    weights_path = os.path.join('new_model.h5')

    if os.path.isfile(weights_path):
        builder.load_weights(weights_path)

    classification_model = builder.get_complete_model(input_shape=(45, 45, 1))

    train_model(model=classification_model, train_gen=train_gen,
                validation_gen=val_gen, m_train=m_train, m_val=m_dev,
                mini_batch_size=mini_batch_size,
                save_path=weights_path, epochs=args.epochs)
