import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication, QClipboard
from PyQt5.QtWebEngine import QtWebEngine
import numpy as np


def pinpoint_digit(pixmap):
    a = pixmap
    left = np.argmax(np.sum(a, axis=0) > 0)
    right = left + np.argmin(np.sum(a, axis=0)[left:] > 0)

    top = np.argmax(np.sum(a, axis=1) > 0)
    bottom = top + np.argmin(np.sum(a, axis=1)[top:] > 0)

    return int(round((top + bottom) / 2.0)), int(round((left + right) / 2))


def extract_x(pixmap, row, col):
    h, w = pixmap.shape
    row = min(h - 14 - 1, max(14, row))
    col = min(w - 14 - 1, max(14, col))
    return (pixmap[row - 14:row + 14, col - 14:col + 14] / 255.0).reshape(1, 28 ** 2)


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


def visualize_slice(x):
    from PIL import Image

    t = np.zeros((28, 28), dtype=np.uint8)
    t[:, :] = (x * 255).reshape(28, 28)
    im = Image.frombytes('L', (28, 28), t.tobytes())
    im.show()


class RecognizedNumber:
    def __init__(self):
        self._digits = ''
        self._locations = []

    def is_power_of(self, number):
        threshold = 28
        dx = number.right_most_x - self.left_most_x
        dy = self.y - number.y

        return dx > -8 and dx < threshold and dy > 20

    @property
    def number(self):
        return int(self._digits)

    @property
    def right_most_x(self):
        return max([x for x, y in self._locations])

    @property
    def left_most_x(self):
        return max([x for x, y in self._locations])

    @property
    def y(self):
        return np.mean(np.array([y for x, y in self._locations]))

    def add(self, digit, x, y):
        self._digits += digit
        self._locations.append((x, y))


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, jobs_queue):
        super().__init__()
        self.jobs_queue = jobs_queue

    def recognize_digits(self, pixmap, model):
        locations = locate_digits(pixmap)

        res = []

        for row, col in locations:
            x = extract_x(pixmap, row, col)
            A = model.predict(x)
            digit = np.argmax(np.max(A, axis=0), axis=0)
            res.append((digit, row, col))

        return res

    def nearest_neighbor(self, digits, x, y):
        filtered = []
        remaining = []
        for digit, row, col in digits:
            if abs(self._phi(x, y, col, row)) < np.pi / 8:
                filtered.append((digit, row, col))
            else:
                remaining.append((digit, row, col))

        def distance(triple):
            digit, row, col = triple
            return self._distance(x, y, col, row)

        sorted_digits = sorted(filtered, key=distance, reverse=True)
        if not sorted_digits:
            return

        first_digit, row, col = sorted_digits.pop()
        if self._are_neighbors(x, y, col, row):
            return first_digit, row, col

    def _are_neighbors(self, x1, y1, x2, y2):
        d = self._distance(x1, y1, x2, y2)
        phi = self._phi(x1, y1, x2, y2)

        return d < 50 and abs(phi) < np.pi / 16

    def _distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _phi(self, x1, y1, x2, y2):
        dy = y2 - y1
        dx = x2 - x1
        epsilon = 10 ** (-8)
        return np.arctan(dy / (dx + epsilon))

    def recognize_number(self, digits):
        remaining = list(digits)
        remaining.reverse()
        digit, row, col = remaining.pop()

        current_number = RecognizedNumber()
        current_number.add(str(digit), col, row)
        while remaining:
            res = self.nearest_neighbor(remaining, col, row)

            if res is None:
                return current_number, remaining

            neighbor, row, col = res
            remaining.remove((neighbor, row, col))
            current_number.add(str(neighbor), col, row)

        return current_number, remaining

    def recognize_numbers(self, digits):
        numbers = []

        rem = list(digits)
        while True:
            sorted_digits = sorted(rem, key=lambda t: (t[2], t[1]))

            number, rem = self.recognize_number(sorted_digits)
            numbers.append(number)
            if not rem:
                return numbers

    def recognize_powers(self, numbers):
        pows = []
        numbers_in_pow = set()
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                a = numbers[i]
                b = numbers[j]
                if a.is_power_of(b):
                    pows.append('{}^{{{}}}'.format(a.number, b.number))
                    numbers_in_pow.add(a.number)
                    numbers_in_pow.add(b.number)

        rest = [str(n.number) for n in numbers if n.number not in numbers_in_pow]

        res = pows + rest
        return ' '.join(res)

    def run(self):
        from models import get_model
        model = get_model()

        while True:
            pixmap = self.jobs_queue.get()

            digits = self.recognize_digits(pixmap, model)

            if not digits:
                continue
            numbers = self.recognize_numbers(digits)

            if numbers:
                pows_string = self.recognize_powers(numbers)
                print(pows_string)
                #str_numbers = [str(num.number) for num in numbers]
                self.completed.emit(pows_string)


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
