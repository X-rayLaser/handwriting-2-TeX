import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine
import numpy as np
from segmentation import extract_segments
from construction import LatexBuilder
from building_blocks import Digit
import config
from dataset_utils import index_to_class


image_size = config.image_size


def recognize(segments, model):
    res = []

    for segment in segments:
        x, y = segment.bounding_box.xy_center

        if segment.bounding_box.width > 60:
            res.append(Digit('div', x, y))
        else:
            input = prepare_input(segment)
            A = model.predict(input)
            class_index = np.argmax(np.max(A, axis=0), axis=0)

            category_class = index_to_class[class_index]
            res.append(Digit(category_class, x, y))

    return res


def prepare_input(segment):
    x = segment.pixels / 255.0
    return x.reshape(1, image_size, image_size, 1)


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, jobs_queue):
        super().__init__()
        self.jobs_queue = jobs_queue

    def construct_latex(self, pixmap, model):
        from construction import construct_latex
        segments = extract_segments(pixmap)

        digits = recognize(segments, model)

        return construct_latex(digits, pixmap.shape[1], pixmap.shape[0])

    def run(self):
        from models import get_math_symbols_model

        model = get_math_symbols_model()

        while True:
            pixmap = self.jobs_queue.get()
            latex_str = self.construct_latex(pixmap, model)

            if latex_str:
                self.completed.emit(latex_str)


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
