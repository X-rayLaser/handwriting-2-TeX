import sys
from queue import Queue
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine


class TuningManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self):
        super().__init__()

    @QtCore.pyqtSlot(list, str)
    def add_image(self, image, label):
        from PIL import Image
        import numpy as np
        a = bytes(image)
        im = Image.frombytes('RGBA', (90, 90), a).convert('LA')
        im.save('canvas.png')
        #im = Image.open('canvas.png')
        pixels = np.array([lum for alpha, lum in list(im.getdata())])
        pixels = pixels.reshape(90, 90)

    @QtCore.pyqtSlot()
    def fine_tune(self):
        pass


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
