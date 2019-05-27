import random
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from dataset_utils import index_to_class
from dataset_utils import load_dataset, shuffle_data
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
        self.output_width = output_width
        self.output_height = output_height

    @property
    def background_class(self):
        return self.num_classes - 1

    def load_raw_examples(self):
        return load_dataset(csv_files_dir=self.csv_files_dir)

    def distort_example(self, x, y):
        x = x.reshape((1, self.output_height, self.output_width, 1))
        for x_batch, y_batch in self.distortions_generator.flow(x, [y], batch_size=1):
            return np.array(x_batch[0], dtype=np.uint8)

    def create_shifted_image(self, x, shift):
        x = x.reshape(self.output_height, self.output_width)

        return create_example_image(x, max_shift=shift)

    def decide_to_detect(self):
        return random.random() > self.p_background

    def preprocess(self, x, y):
        x = self.distort_example(x, y)

        if self.decide_to_detect():
            shift = self.max_shift
        else:
            shift = self.output_height
            y = self.background_class

        x = self.create_shifted_image(x, shift)

        y = to_categorical(y, num_classes=self.num_classes)

        return x, y


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
