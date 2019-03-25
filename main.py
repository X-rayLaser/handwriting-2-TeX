import sys
from PyQt5 import QtCore
from PyQt5.QtCore import QUrl
from PyQt5.QtQml import QQmlApplicationEngine
from PyQt5.QtGui import QGuiApplication, QClipboard
from PyQt5.QtWebEngine import QtWebEngine


class Recognizer(QtCore.QThread):
    completed = QtCore.pyqtSignal(str)

    def __init__(self, pixels):
        super().__init__()
        self.pixels = pixels

    def run(self):
        from models import get_model, normalize
        import numpy as np
        pixmap = self.pixels

        win_size = 28
        height = pixmap.shape[0]
        width = pixmap.shape[1]
        model = get_model()

        window = pixmap[:win_size, :win_size]
        window = window.reshape(28 * 28, 1)
        print(window)
        print(np.max(window))
        print(np.mean(window))


        x = normalize(window)
        digit, prob = model.predict(x)
        print(digit, prob)
        res = str(digit)
        self.completed.emit(res)

        return

        best_match = (0, 0)

        for i in range(height - win_size + 1):
            for j in range(width - win_size + 1):
                window = pixmap[i:i+win_size, j:j+win_size]
                assert window.shape[0] == 28
                assert window.shape[1] == 28
                window = window.reshape(28*28, 1)

                x = normalize(window)
                digit, prob = model.predict(x)
                if prob > best_match[1]:
                    best_match = digit, prob
                    print(best_match)
                print(i, j)

        digit = best_match[0]
        res = str(digit)
        self.completed.emit(res)


class AppManager(QtCore.QObject):
    predictionReady = QtCore.pyqtSignal(str, arguments=['prediction'])

    def __init__(self, clipboard):
        super().__init__()
        self.clipboard = clipboard
        self.thread = None

    @QtCore.pyqtSlot(list, int, int)
    def recognize(self, pixels, width, height):
        from PIL import Image
        import numpy as np
        a = bytes(pixels)
        im = Image.frombytes('RGBA', (width, height), a).convert('LA')
        im.save('canvas.png')
        im = Image.open('canvas.png')
        pixels = np.array([lum for alpha, lum in list(im.getdata())]).reshape(height, width)
        self.thread = Recognizer(pixels)
        thread = self.thread

        thread.completed.connect(lambda res: self.predictionReady.emit(res))

        thread.start()

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
