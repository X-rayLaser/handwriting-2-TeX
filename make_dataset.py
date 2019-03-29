import os
from mnist import MNIST
from util import random_transformation, list_to_line, dataset_root


def extend_training_set(images, labels, on_example_ready):
    for i in range(len(images)):
        image = images[i]
        label = labels[i]

        x = list_to_line(image)
        y = str(label) + '\n'
        on_example_ready(x, y)

        for k in range(3):
            timage = random_transformation(image).reshape(28*28).tolist()
            x = list_to_line(timage)
            y = str(label) + '\n'
            on_example_ready(x, y)

        if i % 1000 == 0:
            print('Created {} training examples'.format(4 * i))


def create_training_set():
    mndata = MNIST(dataset_root)
    images, labels = mndata.load_training()

    extended_x_path = os.path.join(dataset_root, 'extended_X.txt')
    extended_y_path = os.path.join(dataset_root, 'extended_Y.txt')

    if os.path.isfile(extended_x_path):
        os.remove(extended_x_path)

    if os.path.isfile(extended_y_path):
        os.remove(extended_y_path)

    def on_example_ready(xline, yline):
        with open(extended_x_path, 'a') as f:
            f.write(xline)

        with open(extended_y_path, 'a') as f:
            f.write(yline)

    extend_training_set(images, labels, on_example_ready)


def create_test_set():
    mndata = MNIST(dataset_root)
    images_test, labels_test = mndata.load_testing()

    test_x_path = os.path.join(dataset_root, 'extended_Xtest.txt')
    test_y_path = os.path.join(dataset_root, 'extended_Ytest.txt')

    with open(test_x_path, 'a') as f:
        for image in images_test:
            f.write(list_to_line(image))

    with open(test_y_path, 'a') as f:
        lines = '\n'.join([str(label) for label in labels_test])
        f.write(lines)


def create_dataset():
    create_training_set()
    create_test_set()


if __name__ == '__main__':
    create_dataset()
