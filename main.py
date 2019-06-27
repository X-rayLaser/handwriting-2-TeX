import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine
import pipeline
from models import build_classification_model
import os


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, jobs_queue, models_paths, img_width=300, img_height=300):
        super().__init__()
        self.jobs_queue = jobs_queue

        self.img_width = img_width
        self.img_height = img_height

        self._models_paths = models_paths

        self._current_model = 'classification_model.h5'

        self._classifiers = {}

    def get_job(self):
        job = self.jobs_queue.get()

        while not self.jobs_queue.empty():
            job = self.jobs_queue.get()
        return job

    def load_models(self):
        classifiers = {}
        for path, fname in self._models_paths:
            builder = build_classification_model(input_shape=(45, 45, 1), num_classes=14)
            builder.load_weights(path)
            classifiers[fname] = builder.get_complete_model(input_shape=(45, 45, 1))
        return classifiers

    @property
    def classifiers(self):
        return self._classifiers.keys()

    def run(self):
        self._classifiers = self.load_models()

        while True:
            image, classifier_name = self.get_job()
            localization_model = self._classifiers[classifier_name]

            latex_str = pipeline.image_to_latex(image, localization_model)

            if latex_str:
                self.completed.emit(latex_str)


class AppManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self, clipboard):
        super().__init__()
        self.clipboard = clipboard
        self.jobs = Queue()

        model_paths = self.get_model_paths()
        self.thread = Recognizer(self.jobs, model_paths, img_width=400, img_height=300)

        self.thread.completed.connect(
            lambda res: self.predictionReady.emit(res)
        )

        self._classifier_name = 'classification_model.h5'

        self.thread.start()

    def get_model_paths(self):
        paths = []
        models_dir = './'
        for fname in os.listdir(models_dir):
            model_path = os.path.join(models_dir, fname)
            _, extension = os.path.splitext(model_path)
            if os.path.isfile(model_path) and extension == '.h5':
                paths.append((model_path, fname))

        return paths

    @QtCore.pyqtSlot(list, int, int)
    def recognize(self, pixels, width, height):
        from PIL import Image
        import numpy as np
        a = bytes(pixels)
        im = Image.frombytes('RGBA', (width, height), a).convert('LA')
        im.save('canvas.png')
        im = Image.open('canvas.png')
        pixels = np.array([lum for alpha, lum in list(im.getdata())]).reshape(height, width)
        self.jobs.put((pixels, self._classifier_name))

    @QtCore.pyqtSlot(str)
    def copy_to_clipboard(self, text):
        self.clipboard.setText(text)

    @QtCore.pyqtSlot(str)
    def set_classifier(self, classifier_name):
        self._classifier_name = classifier_name

    @QtCore.pyqtProperty(list)
    def classifiers(self):
        return [fname for path, fname in self.get_model_paths()]


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
