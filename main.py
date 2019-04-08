import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication, QClipboard
from PyQt5.QtWebEngine import QtWebEngine
import numpy as np


def calculate_range(pixmap_size, win_size, start_index):
    shift = win_size
    max_index = pixmap_size - win_size
    return (max(0, start_index - shift),
            min(max_index + 1, start_index + shift))


def pixmap_slices(pixmap, i0, j0):
    h, w = pixmap.shape

    win_size = 28
    shift = win_size // 7
    ifrom = max(0, i0 - shift)
    ito = min(h - win_size + 1, i0 + shift)

    jfrom = max(0, j0 - shift)
    jto = min(w - win_size + 1, j0 + shift)

    for i in range(ifrom, ito):
        for j in range(jfrom, jto):
            yield pixmap[i:i + win_size, j:j + win_size]


def pinpoint_digit(pixmap):
    a = pixmap
    left = np.argmax(np.sum(a, axis=0) > 0)
    right = left + np.argmin(np.sum(a, axis=0)[left:] > 0)

    top = np.argmax(np.sum(a, axis=1) > 0)
    bottom = top + np.argmin(np.sum(a, axis=1)[top:] > 0)

    return int(round((top + bottom) / 2.0)), int(round((left + right) / 2))


def contains_digit(pixmap):
    return np.sum(pixmap) > 10000


def extract_x(pixmap, row, col):
    h, w = pixmap.shape
    row = min(h - 14 - 1, max(14, row))
    col = min(w - 14 - 1, max(14, col))
    return (pixmap[row - 14:row + 14, col - 14:col + 14] / 255.0).reshape(1, 28 ** 2)


def locate_digits(pixmap, max_cells):
    h, w = pixmap.shape
    hor_cell_size = int(round(w / max_cells))
    vert_cell_size = int(round(h / max_cells))

    locations = []

    for y in range(0, h, vert_cell_size):
        for x in range(0, w, hor_cell_size):
            slice = pixmap[y:y + vert_cell_size, x:x + hor_cell_size]
            if contains_digit(slice):
                row, col = pinpoint_digit(slice)
                row += y
                col += x
                locations.append((row, col))
    return locations


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


def locate_digits_with_graph_processing(pixmap):
    import networkx as nx
    G = nx.Graph()

    h, w = pixmap.shape
    n = h * w

    def index_to_point(index):
        x = index % w
        y = index // w
        return x, y

    def point_to_index(x, y):
        return y * w + x

    G.add_nodes_from(list(range(n)))

    sm = smooth(pixmap, 0.5)

    for i in range(h):
        for j in range(w):
            plain_index = point_to_index(x=j, y=i)
            pixel = sm[i, j]

            if i != 0:
                pixel_above = sm[i - 1, j]
                plain_index_above = point_to_index(j, i - 1)

                if pixel * pixel_above > 1:
                    G.add_edge(plain_index, plain_index_above)

            if j != 0:
                left_pixel = sm[i, j - 1]
                plain_left_index = point_to_index(j - 1, i)

                if pixel * left_pixel > 1:
                    G.add_edge(plain_index, plain_left_index)

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


def visualize_slice(x):
    from PIL import Image

    t = np.zeros((28, 28), dtype=np.uint8)
    t[:, :] = (x * 255).reshape(28, 28)
    im = Image.frombytes('L', (28, 28), t.tobytes())
    im.show()


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, jobs_queue):
        super().__init__()
        self.jobs_queue = jobs_queue

    def run(self):
        from models import get_model
        model = get_model()

        while True:
            pixmap = self.jobs_queue.get()

            locations = locate_digits_with_graph_processing(pixmap)

            res = ''
            for row, col in locations:
                x = extract_x(pixmap, row, col)
                A = model.predict(x)
                digit = np.argmax(np.max(A, axis=0), axis=0)
                res = res + ' ' + str(digit)

            if res:
                self.completed.emit(res)


class AppManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self, clipboard):
        super().__init__()
        self.clipboard = clipboard
        self.jobs = Queue()
        self.thread = Recognizer(self.jobs)

        self.thread.completed.connect(
            lambda res: self.predictionReady.emit(res)
        )

        self.thread.start()

    @QtCore.pyqtSlot(list, int, int)
    def recognize(self, pixels, width, height):
        from PIL import Image
        import numpy as np
        a = bytes(pixels)
        im = Image.frombytes('RGBA', (width, height), a).convert('LA')
        im.save('canvas.png')
        im = Image.open('canvas.png')
        pixels = np.array([lum for alpha, lum in list(im.getdata())]).reshape(height, width)
        self.jobs.put(pixels)

    @QtCore.pyqtSlot(str)
    def copy_to_clipboard(self, text):
        self.clipboard.setText(text)


if __name__ == '__main__':
    sys_argv = sys.argv
    sys_argv += ['--style', 'Imagine']
    app = QGuiApplication(sys.argv)
    QtWebEngine.initialize()
    engine = QQmlApplicationEngine()

    clipboard = app.clipboard()
    manager = AppManager(clipboard)
    engine.rootContext().setContextProperty("manager", manager)

    engine.load(QUrl("qml/main.qml"))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
