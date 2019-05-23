import random
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from keras.layers import Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from dataset_utils import index_to_class
from dataset_utils import load_dataset, shuffle_data
from dataset_utils import dataset_size
from data_synthesis import augmented_generator


def create_example_image(source_image_data, width=45, height=45,
                         output_width=45, output_height=45, max_shift=45):
    a = source_image_data.reshape(height, width)
    source_image = Image.frombytes('L', (a.shape[1], a.shape[0]), a.tobytes())
    img = Image.frombytes('L', (output_width, output_height),
                          np.zeros((output_height, output_width)).tobytes())

    x0 = random.randint(-max_shift, max_shift)
    y0 = random.randint(-max_shift, max_shift)
    img.paste(source_image, box=(x0, y0, x0 + width, y0 + height))
    return np.array(
        img.getdata(), dtype=np.uint8
    ).reshape(output_height, output_width)


def random_image(image_generator, width=45, height=45, max_shift=45):
    class_index = random.choice(list(range(len(index_to_class))))

    object_class = index_to_class[class_index]
    img = Image.frombytes('L', (width, height), np.zeros((height, width)).tobytes())
    a = np.array(image_generator.next_image(object_class), dtype=np.uint8)
    im = Image.frombytes('L', (a.shape[1], a.shape[0]), a.tobytes())

    x0 = random.randint(-max_shift, max_shift)
    y0 = random.randint(-max_shift, max_shift)
    img.paste(im, box=(x0, y0, x0 + width, y0 + height))

    return np.array(img.getdata(), dtype=np.uint8).reshape(45, 45)


def positive_example(image_generator, width=45, height=45):
    x = random_image(image_generator, width, height, max_shift=5)
    y = 1
    return x, y


def negative_example(image_generator, width=45, height=45):
    x = random_image(image_generator, width, height, max_shift=45)
    y = 0
    return x, y


class MiniBatchGenerator:
    def __init__(self, mini_batch_size, x_shape, y_shape):
        self.mini_batch_size = mini_batch_size
        self.x_shape = x_shape
        self.y_shape = y_shape

    def load_raw_examples(self):
        raise NotImplementedError

    def preprocess(self, x, y):
        raise NotImplementedError

    def prepare_batch(self, x_batch, y_batch):
        x_shape = tuple([self.mini_batch_size]) + self.x_shape
        y_shape = tuple([self.mini_batch_size]) + self.y_shape

        x = np.array(x_batch).reshape(x_shape) / 255
        y = np.array(y_batch).reshape(y_shape)
        return x, y

    def generate(self):
        x_train, y_train = self.load_raw_examples()
        x_batch = []
        y_batch = []

        while True:
            x_train, y_train = shuffle_data(x_train, y_train)

            for i in range(len(y_train)):
                x, y = self.preprocess(x_train[i], y_train[i])

                x_batch.append(x)
                y_batch.append(y)

                if len(y_batch) >= self.mini_batch_size:
                    yield self.prepare_batch(x_batch, y_batch)
                    x_batch = []
                    y_batch = []


class DistortedImagesGenerator(MiniBatchGenerator):
    def __init__(self, csv_files_dir, num_classes, mini_batch_size=32,
                 output_width=45, output_height=45, max_shift=2,
                 p_background=0.1):
        super().__init__(mini_batch_size,
                         x_shape=(output_width, output_height, 1),
                         y_shape=(1, 1, num_classes))

        self.csv_files_dir = csv_files_dir
        self.num_classes = num_classes
        self.max_shift = max_shift
        self.distortions_generator = augmented_generator()
        self.p_background = p_background

    @property
    def background_class(self):
        return self.num_classes - 1

    def load_raw_examples(self):
        return load_dataset(csv_files_dir=self.csv_files_dir)

    def distort_example(self, x, y):
        x = x.reshape((1, 45, 45, 1))
        for x_batch, y_batch in self.distortions_generator.flow(x, [y], batch_size=1):
            return np.array(x_batch[0], dtype=np.uint8)

    def preprocess(self, x, y):
        cat_class = index_to_class[y]
        if cat_class not in ['+', 'times']:
            x = self.distort_example(x, y)

        x = x.reshape(45, 45)

        if random.random() > self.p_background:
            x = create_example_image(x, max_shift=self.max_shift)
        else:
            y = self.background_class
            x = create_example_image(x, max_shift=45)

        y = to_categorical(y, num_classes=self.num_classes)

        return x, y


class ClassificationGenerator(MiniBatchGenerator):
    def __init__(self, csv_files_dir, num_classes, mini_batch_size=32,
                 source_width=45, source_height=45, output_width=60,
                 output_height=60, max_shift=20):
        super().__init__(mini_batch_size, x_shape=(output_width, output_height, 1), y_shape=(1, 1, num_classes))
        self.csv_files_dir = csv_files_dir
        self.output_width = output_width
        self.output_height = output_height
        self.source_width = source_width
        self.source_height = source_height
        self.num_classes = num_classes
        self.max_shift = max_shift

    def load_raw_examples(self):
        return load_dataset(csv_files_dir=self.csv_files_dir)

    def preprocess(self, x, y):
        x = create_example_image(x, max_shift=self.max_shift,
                                 width=self.source_width,
                                 height=self.source_height,
                                 output_width=self.output_width,
                                 output_height=self.output_height)
        y = to_categorical(y, num_classes=self.num_classes)
        return x, y


def model(input_shape=(45, 45, 1), num_classes=1):
    drop_prob = 0.1

    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=10, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=20, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=40, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=80, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=200, kernel_size=(7, 7),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Conv2D(filters=200, kernel_size=(1, 1),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(drop_prob))

    model.add(Conv2D(filters=num_classes, kernel_size=(1, 1),
                     kernel_initializer='he_normal', activation='softmax'))
    return model


if __name__ == '__main__':
    csv_dir = '../datasets/digits_and_operators_csv/train'
    csv_dev = '../datasets/digits_and_operators_csv/dev'

    m_train, _ = dataset_size(csv_dir)
    m_val, _ = dataset_size(csv_dev)

    mini_batch_size = 32

    distorted_generator = DistortedImagesGenerator(csv_dir, num_classes=15,
                                                   mini_batch_size=mini_batch_size)

    validation_generator = DistortedImagesGenerator(csv_dev, num_classes=15,
                                                    mini_batch_size=mini_batch_size)

    localization_model = model(num_classes=15)
    localization_model.compile(optimizer='adam', loss='categorical_crossentropy',
                               metrics=['accuracy'])
    localization_model.fit_generator(
        generator=distorted_generator.generate(),
        steps_per_epoch=int(m_train / mini_batch_size),
        epochs=6,
        validation_data=validation_generator.generate(),
        validation_steps=int(m_val / mini_batch_size)
    )
    localization_model.save_weights('../localization_model.h5')
