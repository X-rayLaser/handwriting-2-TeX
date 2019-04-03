import numpy as np


def augmented_dataset_generator(images, labels, batch_size=32):
    import keras
    from keras.preprocessing.image import ImageDataGenerator

    m = len(images)
    x_train = np.array(images).reshape((m, 28, 28, 1))
    y_train = np.array(labels)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=15,
        width_shift_range=5,
        height_shift_range=5,
        horizontal_flip=False)

    datagen.fit(x_train)
    gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    return gen
