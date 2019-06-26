import sys
import os
import shutil
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine
from models import build_classification_model, train_model
from dataset_utils import dataset_generator, class_to_index
import numpy as np


class Calibrator(QtCore.QThread):
    completed = QtCore.pyqtSignal(float)

    def __init__(self, jobs_queue, img_width=45, img_height=45):
        super().__init__()
        self.jobs_queue = jobs_queue

        self.img_width = img_width
        self.img_height = img_height

    def get_job(self):
        job = self.jobs_queue.get()

        while not self.jobs_queue.empty():
            job = self.jobs_queue.get()
        return job

    def get_generator(self, examples, batch_size=128):
        w = self.img_width
        h = self.img_height
        m = len(examples)
        x = np.zeros((m, h * w))
        y = np.zeros(m)

        for i in range(m):
            pixels, label = examples[i]

            x[i] = pixels.reshape(h * w)
            y[i] = class_to_index[label]

        def wrapped_generator():
            for x_batch, y_batch in dataset_generator(
                    x, y, mini_batch_size=batch_size):
                yield x_batch, y_batch.reshape((-1, 1, 1, 14))

        gen = wrapped_generator()
        return gen

    def run(self):
        tuned_path = 'tuned_model.h5'
        original_model = 'classification_model.h5'

        if not os.path.isfile(tuned_path):
            shutil.copyfile(original_model, tuned_path)

        builder = build_classification_model(input_shape=(45, 45, 1),
                                             num_classes=14)
        builder.load_weights(tuned_path)

        classifier = builder.get_complete_model(input_shape=(45, 45, 1))

        while True:
            examples = self.get_job()
            print(examples)

            total = len(examples)
            batch_size = 128
            m_train = int(round(total * 0.6))
            m_val = total - m_train

            print(total, m_train, m_val)

            from random import shuffle
            from models import calculate_num_steps

            shuffle(examples)

            train_examples = examples[:m_train]
            val_examples = examples[m_train:]

            train_gen = self.get_generator(train_examples, batch_size)
            val_gen = self.get_generator(val_examples, batch_size)

            train_model(model=classifier, train_gen=train_gen,
                        validation_gen=val_gen, m_train=m_train, m_val=m_val,
                        mini_batch_size=batch_size,
                        save_path=tuned_path, epochs=1)

            metrics = classifier.evaluate_generator(
                generator=self.get_generator(val_examples, batch_size),
                steps=calculate_num_steps(m_val, batch_size)
            )

            print(metrics)

            self.completed.emit(metrics[-1])


class TuningManager(QtCore.QObject):
    tuningComplete = QtCore.pyqtSignal(float, arguments=['accuracy'])

    def __init__(self):
        super().__init__()
        self._examples = []
        self._jobs = Queue()
        self._calibrator = Calibrator(self._jobs, img_width=45, img_height=45)

        def handle_completed(val_accuracy):
            self.tuningComplete.emit(val_accuracy)

        self._calibrator.completed.connect(handle_completed)

        self._calibrator.start()

    @QtCore.pyqtSlot(list, str)
    def add_image(self, image, label):
        from PIL import Image
        import numpy as np
        from skimage.transform import resize
        from skimage import img_as_ubyte

        a = bytes(image)
        im = Image.frombytes('RGBA', (90, 90), a).convert('LA')

        pixels = np.array([lum for alpha, lum in list(im.getdata())], dtype=np.uint8)
        pixels = pixels.reshape(90, 90)

        resized = resize(pixels, (45, 45), anti_aliasing=True)
        resized = img_as_ubyte(resized)

        self._examples.append((resized, label))

    @QtCore.pyqtSlot()
    def fine_tune(self):
        self._jobs.put(self._examples)
        self._examples = []


if __name__ == '__main__':
    sys_argv = sys.argv
    sys_argv += ['--style', 'Imagine']
    app = QGuiApplication(sys.argv)
    QtWebEngine.initialize()
    engine = QQmlApplicationEngine()

    manager = TuningManager()
    engine.rootContext().setContextProperty("manager", manager)

    engine.load(QUrl("qml/fine_tune.qml"))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
