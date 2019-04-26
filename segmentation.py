import numpy as np
import config
from building_blocks import RectangularRegion

image_size = config.image_size


def pinpoint_digit(pixmap):
    a = pixmap
    left = np.argmax(np.sum(a, axis=0) > 0)
    right = left + np.argmin(np.sum(a, axis=0)[left:] > 0)

    top = np.argmax(np.sum(a, axis=1) > 0)
    bottom = top + np.argmin(np.sum(a, axis=1)[top:] > 0)

    y = int(round(top))
    x = int(round(left))

    width = right - left
    height = bottom - top
    assert width > 0
    assert height > 0
    return RectangularRegion(x, y, width, height)


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

    bounding_boxes = []
    digit_drawings = [component for component in nx.connected_components(G)
                      if len(component) > 1]

    for drawing in digit_drawings:
        a = np.zeros((h, w), dtype=np.uint8)

        temp = np.zeros(w * h, dtype=np.bool)
        temp[list(drawing)] = True
        a[:, :] = sm * temp.reshape(h, w)

        bounding_boxes.append(pinpoint_digit(a))
    return bounding_boxes


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
    def __init__(self, pixels, bounding_box):
        self.pixels = pixels
        self.bounding_box = bounding_box


def extract_segments(pixmap):
    bounding_boxes = locate_digits(pixmap)

    segments = []
    for box in bounding_boxes:
        x, y = box.xy_center
        canvas_slice = extract_segment_center(pixmap, int(y), int(x))
        segment = UnidentifiedObject(canvas_slice, box)
        segments.append(segment)

    return segments


def visualize_slice(x):
    from PIL import Image

    t = np.zeros((image_size, image_size), dtype=np.uint8)
    t[:, :] = (x * 255).reshape(image_size, image_size)
    im = Image.frombytes('L', (image_size, image_size), t.tobytes())
    im.show()


def extract_segment(pixmap, bounding_box):
    x0 = int(round(bounding_box.x - bounding_box.width / 2.0))
    y0 = int(round(bounding_box.y - bounding_box.height / 2.0))
    x = x0 + bounding_box.width
    y = y0 + bounding_box.height

    return pixmap[y0:y, x0:x]


def extract_segment_center(pixmap, row, col):
    h, w = pixmap.shape

    half_size = image_size // 2

    row = min(h - half_size - 1, max(half_size, row))
    col = min(w - half_size - 1, max(half_size, col))

    if half_size * 2 < image_size:
        delta = 1
    else:
        delta = 0
    return pixmap[row - half_size:row + half_size + delta, col - half_size:col + half_size + delta]
