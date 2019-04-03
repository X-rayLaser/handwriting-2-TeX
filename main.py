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
    col = np.argmax(np.sum(a, axis=0))
    row = np.argmax(np.sum(a, axis=1))
    return row, col


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

            row, col = pinpoint_digit(pixmap)
            print(row, col)
            X = np.zeros((9**2, 28**2))
            i = 0
            for window in pixmap_slices(pixmap, row, col):
                x = window / 255.0
                X[i, :] = x.reshape(1, 28**2)
                i += 1

            A = model.predict(X)
            res = np.argmax(np.max(A, axis=0))
            self.completed.emit(str(res))


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
