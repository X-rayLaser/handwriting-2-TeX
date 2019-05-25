from random import choice
import numpy as np
from dataset_utils import index_to_class, class_to_index
from data_synthesis import Canvas
from keras.utils import to_categorical


def uniform_flow(csv_dir, num_classes, image_width, image_height, batch_size=32):
    canvas = Canvas(width=image_width, height=image_height,
                    dataset_dir=csv_dir)

    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            canvas.reset()
            ground_true = choice(index_to_class)

            canvas.draw_random_class_image(0, 0, ground_true)
            image_data = canvas.image_data

            x = image_data
            y = class_to_index[ground_true]
            y = to_categorical(y, num_classes=num_classes)
            x_batch.append(x)
            y_batch.append(y)

        x_batch = np.array(x_batch).reshape((batch_size, image_height, image_width, 1))
        x_batch = x_batch / 255.0

        y_batch = np.array(y_batch).reshape(batch_size, num_classes)

        yield x_batch, y_batch
