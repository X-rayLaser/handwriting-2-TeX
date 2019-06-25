import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine
import pipeline


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, jobs_queue, img_width=300, img_height=300):
        super().__init__()
        self.jobs_queue = jobs_queue

        self.img_width = img_width
        self.img_height = img_height

    def get_job(self):
        job = self.jobs_queue.get()

        while not self.jobs_queue.empty():
            job = self.jobs_queue.get()
        return job

    def run(self):
        from models import build_classification_model

        builder = build_classification_model(input_shape=(45, 45, 1), num_classes=14)
        #builder.load_weights('classification_model.h5')
        builder.load_weights('new_model.h5')
        localization_model = builder.get_complete_model(input_shape=(45, 45, 1))

        while True:
            image = self.get_job()
            latex_str = pipeline.image_to_latex(image, localization_model)

            if latex_str:
                self.completed.emit(latex_str)


class AppManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self, clipboard):
        super().__init__()
        self.clipboard = clipboard
        self.jobs = Queue()
        self.thread = Recognizer(self.jobs, img_width=400, img_height=300)

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
