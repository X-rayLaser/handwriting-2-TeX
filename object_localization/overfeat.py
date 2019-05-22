import numpy as np
from PIL import Image
from keras.layers import Input, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from keras import Model


def create_model(image_width, image_height, num_classes):
    input_shape = (image_height, image_width, 1)
    fc_kernel_height = image_height
    fc_kernel_width = image_width
    drop_prob = 0.1

    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=6, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    fc_kernel_height = (fc_kernel_height - 2) // 2
    fc_kernel_width = (fc_kernel_width - 2) // 2

    model.add(Conv2D(filters=12, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    fc_kernel_height = (fc_kernel_height - 2) // 2
    fc_kernel_width = (fc_kernel_width - 2) // 2

    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=48, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu',
                     name='last_feature_extractor'))
    model.add(BatchNormalization())
    model.add(MaxPool2D(pool_size=(2, 2)))

    fc_kernel_height = (fc_kernel_height - 4) // 2
    fc_kernel_width = (fc_kernel_width - 4) // 2

    model.add(Conv2D(filters=100, kernel_size=(fc_kernel_height, fc_kernel_width),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Conv2D(filters=100, kernel_size=(1, 1),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Conv2D(filters=num_classes, kernel_size=(1, 1),
                     kernel_initializer='he_normal', activation='softmax'))
    return model


class ModelBuilder:
    def __init__(self, input_shape, initial_num_filters=6):
        self._filters = initial_num_filters
        self._model = Sequential()
        self._model.add(Input(shape=input_shape))
        h, w, d = input_shape
        self._input_width = w
        self._input_height = h

    def add_conv_layer(self, last_one=False):
        kwargs = dict(filters=self._filters, kernel_size=(3, 3),
                      kernel_initializer='he_normal', activation='relu')
        if last_one:
            kwargs['name'] ='last_feature_extraction_layer'

        layer = Conv2D(**kwargs)

        self._model.add(layer)
        self._filters *= 2
        self._input_width -= 2
        self._input_height -= 2

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

    @property
    def complete_model(self):
        return self._model

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


def build_model(input_shape, num_classes):
    builder = ModelBuilder(input_shape)
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer(last_one=True).add_pooling_layer().add_batch_norm_layer()
    builder.add_fully_connected_layer().add_fully_connected_layer()
    builder.add_output_layer(num_classes=num_classes)

    return builder


def build_feature_extractor(input_shape):
    builder = ModelBuilder(input_shape)
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    return builder.model


def build_classifier(input_shape):


def simple_model():
    layers = ['conv', 'conv', 'pool', 'conv', 'conv', 'pool']


def get_scales(size, min_size, scales):
    min_scale = min_size / size
    eps = 0.01
    step_size = (1 - min_scale) / scales - eps

    scale = 1
    koefficients = []
    while scale > min_scale:
        koefficients.append(scale)
        scale -= step_size

    return koefficients


def rescaled_images(image_array, min_size=80, scales=10):
    original_height, original_width = image_array

    for scale in get_scales(original_height, min_size, scales):
        scaled_width = int(round(original_width * scale))
        scaled_height = int(round(original_height * scale))

        image = Image.frombytes('L', (original_width, original_height),
                                image_array)

        image = image.resize((scaled_width, scaled_height))
        yield np.array(image.getdata(), dtype=np.uint8).reshape(
            scaled_height, scaled_width
        )


def get_feature_extractor(model, input_shape):
    from keras.layers import Input
    from keras import Model
    inp = Input(shape=input_shape)
    x = inp
    for layer in model.layers:
        if layer.name == 'last_feature_extractor':
            print('Found it!')
            break

        x = layer(x)

    out = x
    return Model(input=inp, output=out)


def get_classifier(model, input_shape):
    inp = Input(shape=input_shape)
    x = inp

    index = -1
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if layer.name == 'last_feature_extractor':
            index = i + 1
            break

    for i in range(index, len(model.layers)):
        layer = model.layers[i]
        x = layer(x)

    out = x
    return Model(input=inp, output=out)


def augmented_volumes(unpooled_volume, pool_size=2):
    height, width, depth = unpooled_volume.shape

    for i in range(pool_size):
        for j in range(pool_size):
            unpooled_volume[i:, j:, :]


def classify(image_array, min_size=80, scales=10):
    original_height, original_width = image_array

    class_distibutions = []
    for img_array in rescaled_images(image_array, min_size, scales):
        h, w = img_array.shape
        input_shape = (h, w, 1)
        batch_shape = (1,) + input_shape

        builder = build_model(input_shape, num_classes=14)

        feature_extractor = builder.get_feature_extractor(input_shape)

        x = img_array.reshape(batch_shape)
        x /= 255
        unpooled_volume = feature_extractor.predict(x)

        classifier = builder.get_classifier(input_shape=unpooled_volume.shape)
        output = classifier.predict(unpooled_volume)[0]
        classes = output.max(axis=(0, 1))
        class_distibutions.append(classes)

    average = np.array(class_distibutions).mean(axis=0)
    from dataset_utils import index_to_class

    index = average.argmax()
    return index_to_class[index]
