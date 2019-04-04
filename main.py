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
    return (pixmap[row - 14:row + 14, col - 14:col + 14] / 255.0).reshape(1, 28 ** 2)


def locate_digits(pixmap):
    h, w = pixmap.shape
    midy = int(round(h / 2))
    midx = int(round(w / 2))
    top_left = pixmap[:midy, :midx]
    top_right = pixmap[:midy, midx:]
    bottom_left = pixmap[midy:, :midx]
    bottom_right = pixmap[midy:, midx:]

    locations = []
    if contains_digit(top_left):
        row, col = pinpoint_digit(top_left)
        locations.append((row, col))

    if contains_digit(top_right):
        row, col = pinpoint_digit(top_right)
        col += midx
        locations.append((row, col))

    if contains_digit(bottom_left):
        row, col = pinpoint_digit(bottom_left)
        row += midy
        locations.append((row, col))

    if contains_digit(bottom_right):
        row, col = pinpoint_digit(bottom_right)
        row += midy
        col += midx
        locations.append((row, col))

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

            locations = locate_digits(pixmap)

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
