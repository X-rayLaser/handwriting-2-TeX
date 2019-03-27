import os
import numpy as np


dataset_root = './datasets/mnist'


def shift_pixels_right(pixels, k):
    image = np.array(pixels).reshape(28, 28)
    x = np.empty_like(image)
    x[:, k:28] = image[:, :28-k]
    x[:, :k] = np.zeros((28, k))
    return x.reshape(28*28).tolist()


def shift_pixels_left(pixels, k):
    image = np.array(pixels).reshape(28, 28)
    x = np.empty_like(image)
    x[:, :28-k] = image[:, k:28]

    x[:, 28-k:] = np.zeros((28, k))
    return x.reshape(28*28).tolist()


def horizontal_shift(pixels, k):
    if k > 0:
        return shift_pixels_right(pixels, k)
    else:
        return shift_pixels_left(pixels, abs(k))


def vertical_shift(pixels, k):
    if k > 0:
        return shift_pixels_down(pixels, k)
    else:
        return shift_pixels_up(pixels, k)


def shift_pixels_up(pixels, k):
    image = np.array(pixels).reshape(28, 28)

    x = np.empty_like(image)
    x[:k, :] = image[-k:, :]
    x[-k:, :] = np.zeros((k, 28))
    return x.reshape(28*28).tolist()


def shift_pixels_down(pixels, k):
    image = np.array(pixels).reshape(28, 28)

    x = np.empty_like(image)
    x[:28-k, :] = image[k:28, :]

    x[28-k:, :] = np.zeros((k, 28))
    return x.reshape(28*28).tolist()


def embed_noise(a, noise=50):
    assert a.ndim == 2
    res = np.zeros_like(a, dtype=np.int)
    res[:, :] = np.minimum(np.random.randn(*a.shape) * noise + a, 255)
    return np.maximum(res, 0)


class CoordinateSystem:
    def __init__(self, a, x0=0, y0=0):
        self.a = a
        self.x0 = x0
        self.y0 = y0

    def map_to_coordinates(self):
        return map_to_coordinates(self.a, self.x0, self.y0)

    def coordinates_to_cell(self, x, y):
        row, col = coordinates_to_cell(x, y, self.x0, self.y0)
        row = self.a.shape[0] - 1 - row
        return row, col


def map_to_coordinates(a, x0=0, y0=0):
    assert a.ndim == 2

    indices = []
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            x = j - x0
            y = a.shape[0] - 1 - i
            y -= y0
            indices.append((x, y))

    return np.array(indices, dtype=a.dtype).T


def coordinates_to_cell(x, y, x0, y0):
    row = y + y0
    col = x + x0
    return row, col


def within_bounds(size, index):
    return index >= 0 and index < size


def rotate(a, angle, origin=None):
    assert a.ndim == 2
    rads = angle * np.pi / 180.0

    if not origin:
        origin = (a.shape[0] - 1, 0)

    y0 = a.shape[0] - 1 - origin[0]
    x0 = origin[1]

    system = CoordinateSystem(a, x0, y0)

    cos_phi = np.cos(rads)
    sin_phi = np.sin(rads)
    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])

    X = system.map_to_coordinates()

    X_transformed = np.zeros((2, X.shape[1]), dtype=np.int)
    X_transformed[:, :] = np.round(np.dot(R, X))

    res = np.zeros_like(a)
    for j in range(X_transformed.shape[1]):
        pixel_val = a.reshape(a.shape[0] * a.shape[0])[j]

        x = X_transformed[0, j]
        y = X_transformed[1, j]

        row, col = system.coordinates_to_cell(x, y)

        if within_bounds(a.shape[0], row) and within_bounds(a.shape[1], col):
            res[row, col] = pixel_val

    return res


def random_transformation(image):
    max_shift = 6
    hor_shift = np.random.randint(-max_shift, max_shift)
    vert_shift = np.random.randint(-max_shift, max_shift)
    noise_mag = np.random.randint(0, 50)
    rotation_angle = np.random.randint(-10, 10)
    scaling_factor = np.random.randint(1, 3)

    if hor_shift != 0:
        image = horizontal_shift(image, hor_shift)

    if vert_shift != 0:
        image = vertical_shift(image, vert_shift)

    embed_noise(image, noise_mag)
    #image = rotate(image, rotation_angle)
    #image = scale(image, scaling_factor)

    return image


def list_to_line(elements):
    return ' '.join([str(x) for x in elements]) + '\n'


def create_training_set():
    from mnist import MNIST

    mndata = MNIST(dataset_root)
    images, labels = mndata.load_training()

    extended_x_path = os.path.join(dataset_root, 'extended_X.txt')
    extended_y_path = os.path.join(dataset_root, 'extended_Y.txt')

    with open(extended_x_path, 'w') as fx, open(extended_y_path, 'w') as fy:
        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            fx.write(list_to_line(image))
            fy.write(label + '\n')

            for k in range(3):
                timage = random_transformation(image).tolist()
                fx.write(list_to_line(timage))
                fy.write(label + '\n')


def create_test_set():
    from mnist import MNIST
    mndata = MNIST(dataset_root)
    images_test, labels_test = mndata.load_testing()

    test_x_path = os.path.join(dataset_root, 'extended_X.txt')
    test_y_path = os.path.join(dataset_root, 'extended_Y.txt')

    with open(test_x_path, 'w') as f:
        for image in images_test:
            f.write(list_to_line(image))

    with open(test_y_path, 'w') as f:
        lines = '\n'.join([str(label) for label in labels_test])
        f.write(lines)


def create_dataset():
    create_training_set()
    create_test_set()


def train_data_iterator():
    pass
