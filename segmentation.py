import numpy as np


def pinpoint_digit(pixmap):
    a = pixmap
    left = np.argmax(np.sum(a, axis=0) > 0)
    right = left + np.argmin(np.sum(a, axis=0)[left:] > 0)

    top = np.argmax(np.sum(a, axis=1) > 0)
    bottom = top + np.argmin(np.sum(a, axis=1)[top:] > 0)

    return int(round((top + bottom) / 2.0)), int(round((left + right) / 2))


def smooth(a, beta_p=0.9):
    res = np.zeros_like(a)
    beta = np.ones(a.shape[0]) * beta_p
    for i in range(a.shape[0]):
        prev = 0
        for j in range(a.shape[1]):
            prev = beta[i] * prev + (1 - beta[i]) * a[i, j]
            res[i, j] = prev

    for j in range(a.shape[1]):
        prev = 0

        for i in range(a.shape[0]):
            prev = beta[i] * prev + (1 - beta[i]) * res[i, j]
            res[i, j] = prev

    return res


def locate_digits(pixmap):
    import networkx as nx

    h, w = pixmap.shape

    sm = smooth(pixmap, 0.3)

    G = build_graph(sm)

    locations = []
    digit_drawings = [component for component in nx.connected_components(G)
                      if len(component) > 1]

    for drawing in digit_drawings:
        a = np.zeros((h, w), dtype=np.uint8)

        temp = np.zeros(w * h, dtype=np.bool)
        temp[list(drawing)] = True
        a[:, :] = sm * temp.reshape(h, w)

        locations.append(pinpoint_digit(a))
    return locations


def build_graph(pixel_matrix):
    import networkx as nx
    G = nx.Graph()

    h, w = pixel_matrix.shape
    n = h * w

    def point_to_index(x, y):
        return y * w + x

    G.add_nodes_from(list(range(n)))

    for i in range(h):
        for j in range(w):
            plain_index = point_to_index(x=j, y=i)
            pixel = pixel_matrix[i, j]

            if i != 0:
                pixel_above = pixel_matrix[i - 1, j]
                plain_index_above = point_to_index(j, i - 1)

                if pixel * pixel_above > 1:
                    G.add_edge(plain_index, plain_index_above)

            if j != 0:
                left_pixel = pixel_matrix[i, j - 1]
                plain_left_index = point_to_index(j - 1, i)

                if pixel * left_pixel > 1:
                    G.add_edge(plain_index, plain_left_index)

    return G


class UnidentifiedObject:
    def __init__(self, pixels, x, y):
        self.pixels = pixels
        self.x = x
        self.y = y


def extract_segments(pixmap):
    locations = locate_digits(pixmap)

    boxes = []
    for row, col in locations:
        box = extract_box(pixmap, row, col)
        segment = UnidentifiedObject(box, x=col, y=row)
        boxes.append(segment)

    return boxes


def visualize_slice(x):
    from PIL import Image

    t = np.zeros((28, 28), dtype=np.uint8)
    t[:, :] = (x * 255).reshape(28, 28)
    im = Image.frombytes('L', (28, 28), t.tobytes())
    im.show()


def extract_box(pixmap, row, col):
    h, w = pixmap.shape
    row = min(h - 14 - 1, max(14, row))
    col = min(w - 14 - 1, max(14, col))
    return pixmap[row - 14:row + 14, col - 14:col + 14]
