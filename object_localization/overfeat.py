import numpy as np
from PIL import Image
from keras.layers import Input, InputLayer, Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from keras import Model
from dataset_utils import dataset_size
from object_localization.localization_training import ClassificationGenerator


class ModelBuilder:
    def __init__(self, input_shape, initial_num_filters=6):
        self._filters = initial_num_filters
        self._model = Sequential()
        self._model.add(InputLayer(input_shape=input_shape))
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


def build_model(input_shape, num_classes):
    builder = ModelBuilder(input_shape)
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_pooling_layer().add_batch_norm_layer()
    builder.add_conv_layer().add_batch_norm_layer()
    builder.add_conv_layer(last_one=True).add_pooling_layer().add_batch_norm_layer()
    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer(drop_prob=0.4)
    builder.add_fully_connected_layer().add_batch_norm_layer().add_dropout_layer(drop_prob=0.4)
    builder.add_output_layer(num_classes=num_classes)

    return builder


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
    print(image_array.shape)
    original_height, original_width = image_array.shape

    for scale in get_scales(original_height, min_size, scales):
        scaled_width = int(round(original_width * scale))
        scaled_height = int(round(original_height * scale))

        image = Image.frombytes('L', (original_width, original_height),
                                image_array)

        image = image.resize((scaled_width, scaled_height))
        yield np.array(image.getdata(), dtype=np.uint8).reshape(
            scaled_height, scaled_width
        )


def augmented_volumes(unpooled_volume, pool_size=2):
    height, width, depth = unpooled_volume.shape

    for i in range(pool_size):
        for j in range(pool_size):
            unpooled_volume[i:, j:, :]


def classify(image_array, model_builder, min_size=80, scales=1):
    class_distibutions = []
    for img_array in rescaled_images(image_array, min_size, scales):
        h, w = img_array.shape
        input_shape = (h, w, 1)
        batch_shape = (1,) + input_shape

        x = img_array.reshape(batch_shape)
        x = x / 255
        print(x.shape)
        model = model_builder.get_complete_model(input_shape=input_shape)
        y_hat = model.predict(x)[0]
        classes = y_hat.max(axis=(0, 1))
        print(classes.shape)
        print(classes)
        class_distibutions.append(classes)
        continue

        feature_extractor = model_builder.get_feature_extractor(input_shape)


        unpooled_volume = feature_extractor.predict(x)

        print(unpooled_volume.shape)
        classifier = model_builder.get_classifier(input_shape=unpooled_volume.shape[1:])
        output = classifier.predict(unpooled_volume)[0]
        classes = output.max(axis=(0, 1))
        class_distibutions.append(classes)

    average = np.array(class_distibutions).mean(axis=0)
    from dataset_utils import index_to_class

    index = average.argmax()
    res = index_to_class[index]
    print(res)
    return res


def train_model(save_path='../overfeat.h5'):
    csv_dir = '../datasets/digits_and_operators_csv/train'
    csv_dev_dir = '../datasets/digits_and_operators_csv/dev'

    m_train, _ = dataset_size(csv_dir)
    m_val, _ = dataset_size(csv_dev_dir)

    mini_batch_size = 32

    output_width = 60
    output_height = 60

    generator = ClassificationGenerator(csv_dir,
                                        mini_batch_size=mini_batch_size,
                                        num_classes=14,
                                        output_width=output_width,
                                        output_height=output_height)

    dev_generator = ClassificationGenerator(csv_dev_dir,
                                            mini_batch_size=mini_batch_size,
                                            num_classes=14,
                                            output_width=output_width,
                                            output_height=output_height)

    input_shape = (output_height, output_width, 1)
    builder = build_model(input_shape, num_classes=14)
    model = builder.get_complete_model(input_shape=input_shape)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit_generator(
        generator=generator.generate(),
        steps_per_epoch=int(m_train / mini_batch_size),
        epochs=5, validation_data=dev_generator.generate(),
        validation_steps=int(m_val / mini_batch_size)
    )

    model.save_weights(save_path)


if __name__ == '__main__':
    train_model()
