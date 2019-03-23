import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication


class AppManager(QtCore.QObject):
    stateChanged = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    sys_argv = sys.argv
    sys_argv += ['--style', 'Imagine']
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    manager = AppManager()
    engine.rootContext().setContextProperty("manager", manager)

    engine.load(QUrl("qml/main.qml"))

    if not engine.rootObjects():
        sys.exit(-1)

    sys.exit(app.exec_())
