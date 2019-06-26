import sys
import os
import shutil
from queue import Queue
from random import shuffle

import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte

from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine

from models import build_classification_model, train_model, calculate_num_steps
from dataset_utils import dataset_generator, class_to_index


class Calibrator(QtCore.QThread):
    completed = QtCore.pyqtSignal(float)

    SPLIT_RATIO = 0.6

    BATCH_SIZE = 128

    NUM_CLASSES = 14

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
                yield x_batch, y_batch.reshape((-1, 1, 1, self.NUM_CLASSES))

        gen = wrapped_generator()
        return gen

    def run(self):
        w = self.img_width
        h = self.img_height

        input_shape = (h, w, 1)

        tuned_path = 'tuned_model.h5'
        original_model = 'classification_model.h5'

        if not os.path.isfile(tuned_path):
            shutil.copyfile(original_model, tuned_path)

        builder = build_classification_model(input_shape=input_shape,
                                             num_classes=self.NUM_CLASSES)
        builder.load_weights(tuned_path)

        classifier = builder.get_complete_model(input_shape=input_shape)

        while True:
            examples, epochs = self.get_job()

            total = len(examples)
            m_train = int(round(total * self.SPLIT_RATIO))
            m_val = total - m_train

            shuffle(examples)

            train_examples = examples[:m_train]
            val_examples = examples[m_train:]

            train_gen = self.get_generator(train_examples, self.BATCH_SIZE)
            val_gen = self.get_generator(val_examples, self.BATCH_SIZE)

            train_model(model=classifier, train_gen=train_gen,
                        validation_gen=val_gen, m_train=m_train, m_val=m_val,
                        mini_batch_size=self.BATCH_SIZE,
                        save_path=tuned_path, epochs=epochs)

            metrics = classifier.evaluate_generator(
                generator=self.get_generator(val_examples, self.BATCH_SIZE),
                steps=calculate_num_steps(m_val, self.BATCH_SIZE)
            )

            self.completed.emit(metrics[-1])


class TuningManager(QtCore.QObject):
    tuningComplete = QtCore.pyqtSignal(float, arguments=['accuracy'])

    TARGET_WIDTH = 45
    TARGET_HEIGHT = 45

    def __init__(self):
        super().__init__()
        self._examples = []
        self._jobs = Queue()
        self._calibrator = Calibrator(self._jobs, img_width=self.TARGET_WIDTH,
                                      img_height=self.TARGET_HEIGHT)

        def handle_completed(val_accuracy):
            self.tuningComplete.emit(val_accuracy)

        self._calibrator.completed.connect(handle_completed)

        self._calibrator.start()

    @QtCore.pyqtSlot(list, str, int, int)
    def add_image(self, image, label, height, width):
        a = bytes(image)
        im = Image.frombytes('RGBA', (width, height), a).convert('LA')

        pixels = np.array([lum for alpha, lum in list(im.getdata())],
                          dtype=np.uint8)
        pixels = pixels.reshape(height, width)

        resized = resize(pixels, (self.TARGET_HEIGHT, self.TARGET_WIDTH),
                         anti_aliasing=True)
        resized = img_as_ubyte(resized)

        self._examples.append((resized, label))

    @QtCore.pyqtSlot(int)
    def fine_tune(self, epochs):
        self._jobs.put((self._examples, epochs))
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
