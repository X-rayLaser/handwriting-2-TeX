from keras.layers import Input, InputLayer, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from keras import Model


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


def train_model(model, train_gen, validation_gen, m_train, m_val, mini_batch_size=32,
                loss='categorical_crossentropy', metrics=None,
                save_path='../trained_model.h5', epochs=6):
    if metrics is None:
        metrics = ['accuracy']

    model_to_train = model
    model_to_train.compile(optimizer='adam', loss=loss, metrics=metrics)

    model_to_train.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(m_train / mini_batch_size),
        epochs=epochs,
        validation_data=validation_gen,
        validation_steps=int(m_val / mini_batch_size)
    )
    model_to_train.save_weights(save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )

    parser.add_argument('--epochs', type=int, default=8,
                        help='number of iterations')

    args = parser.parse_args()

    #train_math_recognition_model(epochs=args.epochs)
