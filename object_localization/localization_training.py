from keras.utils import to_categorical
from keras.layers import Dropout, BatchNormalization, Conv2D, MaxPool2D
from keras.models import Sequential
from dataset_utils import load_dataset
from dataset_utils import dataset_size
from object_localization.common import MiniBatchGenerator, create_example_image, DistortedImagesGenerator


class ClassificationGenerator(DistortedImagesGenerator):
    def __init__(self, csv_files_dir, mini_batch_size=32, output_width=45,
                 output_height=45, max_shift=2):
        super().__init__(csv_files_dir=csv_files_dir, num_classes=14,
                         mini_batch_size=mini_batch_size,
                         output_width=output_width,
                         output_height=output_height,
                         max_shift=max_shift,
                         p_background=0.001)

    def load_raw_examples(self):
        return load_dataset(csv_files_dir=self.csv_files_dir)

    def preprocess(self, x, y):
        x = self.distort_example(x, y)
        x = self.create_shifted_image(x, self.max_shift)
        y = to_categorical(y, num_classes=self.num_classes)
        return x, y


def model(input_shape=(45, 45, 1), num_classes=1):
    drop_prob = 0.1

    model = Sequential()
    model.add(Conv2D(input_shape=input_shape, filters=6, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=12, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=24, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=48, kernel_size=(3, 3),
                     kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=100, kernel_size=(7, 7),
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


def build_classification_model(input_shape, num_classes):
    from object_localization.overfeat import ModelBuilder

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


if __name__ == '__main__':
    from object_localization.common import train_model
    csv_dir = '../datasets/digits_and_operators_csv/train'
    csv_dev = '../datasets/digits_and_operators_csv/dev'

    m_train, _ = dataset_size(csv_dir)
    m_val, _ = dataset_size(csv_dev)

    mini_batch_size = 32
    image_height = 45
    image_width = 45

    distorted_generator = ClassificationGenerator(csv_dir,
                                                  mini_batch_size=mini_batch_size)

    builder = build_classification_model(input_shape=(image_height, image_width, 1), num_classes=14)
    localization_model = builder.get_complete_model(input_shape=(image_height, image_width, 1))

    validation_generator = ClassificationGenerator(csv_dev,
                                                   mini_batch_size=mini_batch_size)

    train_model(localization_model, train_gen=distorted_generator.generate(),
                validation_gen=validation_generator.generate(),
                m_train=m_train, m_val=m_val, mini_batch_size=mini_batch_size,
                save_path='../localization_model.h5',
                epochs=8)
