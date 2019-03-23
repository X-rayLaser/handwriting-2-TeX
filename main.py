import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtWebEngine import QtWebEngine


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def run(self):
        import time
        time.sleep(1)
        res = '\\\\frac{A}{B + 4}'
        self.completed.emit(res)


class AppManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self):
        super().__init__()
        self.thread = None

    @QtCore.pyqtSlot(list)
    def recognize(self, pixels):
        self.thread = Recognizer()
        thread = self.thread

        thread.completed.connect(lambda res: self.predictionReady.emit(res))

        thread.start()


if __name__ == '__main__':
    sys_argv = sys.argv
    sys_argv += ['--style', 'Imagine']
    app = QGuiApplication(sys.argv)
    QtWebEngine.initialize()
    engine = QQmlApplicationEngine()

    manager = AppManager()
    engine.rootContext().setContextProperty("manager", manager)

    engine.load(QUrl("qml/main.qml"))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
