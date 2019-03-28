import os
import numpy as np


dataset_root = './datasets/mnist'


def shift_pixels_right(image, k):
    x = np.empty_like(image)
    x[:, k:28] = image[:, :28-k]
    x[:, :k] = np.zeros((28, k))
    return x


def shift_pixels_left(image, k):
    x = np.empty_like(image)
    x[:, :28-k] = image[:, k:28]

    x[:, 28-k:] = np.zeros((28, k))
    return x


def shift_pixels_up(image, k):
    x = np.empty_like(image)
    x[:k, :] = image[-k:, :]
    x[-k:, :] = np.zeros((k, 28))
    return x


def shift_pixels_down(image, k):
    x = np.empty_like(image)
    x[:28-k, :] = image[k:28, :]

    x[28-k:, :] = np.zeros((k, 28))
    return x


def horizontal_shift(image, k):
    if k > 0:
        return shift_pixels_right(image, k)
    else:
        return shift_pixels_left(image, abs(k))


def vertical_shift(image, k):
    if k > 0:
        return shift_pixels_down(image, k)
    else:
        return shift_pixels_up(image, abs(k))


def embed_noise(a, noise=50):
    assert a.ndim == 2
    res = np.zeros_like(a, dtype=np.int)
    res[:, :] = np.minimum(np.random.randn(*a.shape) * noise + a, 255)
    return np.maximum(res, 0)


class CoordinateSystem:
    def __init__(self, a, x0=0, y0=0):
        assert a.ndim == 2

        self.a = a
        self.x0 = x0
        self.y0 = y0

    def map_to_coordinates(self):
        a = self.a

        indices = []
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                x = j - self.x0
                y = a.shape[0] - 1 - i
                y -= self.y0
                indices.append((x, y))

        return np.array(indices, dtype=a.dtype).T

    def coordinates_to_cell(self, x, y):
        row = y + self.y0
        col = x + self.x0
        row = self.a.shape[0] - 1 - row

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


def scale(image, scaling_factor):
    return image


def random_transformation(pixels):
    max_shift = 6
    hor_shift = np.random.randint(-max_shift, max_shift)
    vert_shift = np.random.randint(-max_shift, max_shift)
    noise_mag = np.random.randint(15, 30)
    rotation_angle = np.random.randint(-50, 50)
    scaling_factor = np.random.randint(1, 3)

    image = np.array(pixels).reshape(28, 28)

    image = rotate(image, rotation_angle, origin=(14, 14))

    if hor_shift != 0:
        image = horizontal_shift(image, hor_shift)

    if vert_shift != 0:
        image = vertical_shift(image, vert_shift)

    image = embed_noise(image, noise_mag)

    image = scale(image, scaling_factor)

    return image


def list_to_line(elements):
    return ' '.join([str(x) for x in elements]) + '\n'


def line_to_list(line):
    return [int(num) for num in line.strip().split(' ')]


def training_batches(extended_x_path, extended_y_path, batch_size=100):
    X = []
    Y = []
    with open(extended_x_path, 'r') as fx, open(extended_y_path, 'r') as fy:
        while True:
            line = fx.readline()
            if line == '':
                X_matrix = np.array(X, dtype=np.uint8)
                Y_matrix = np.array(Y, dtype=np.uint8).reshape(1, len(Y))
                yield X_matrix.T, Y_matrix
                return
            x = line_to_list(line)
            y = line_to_list(fy.readline())[0]
            X.append(x)
            Y.append(y)
            if len(Y) >= batch_size:
                X_matrix = np.array(X, dtype=np.uint8)
                Y_matrix = np.array(Y, dtype=np.uint8).reshape(1, len(Y))
                yield X_matrix.T, Y_matrix
                X[:] = []
                Y[:] = []
