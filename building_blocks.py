import numpy as np


class CaptionBox:
    def __init__(self, text):
        self.text = text

    def concatenate(self, box):
        pass

    @property
    def latex(self):
        return self.text


class Digit:
    def __init__(self, digit, x, y):
        self.digit = digit
        self.x = x
        self.y = y


class RecognizedNumber:
    def __init__(self):
        self._digits = ''
        self._locations = []

    def is_power_of(self, number):
        threshold = 28
        dx = number.left_most_x - self.right_most_x
        dy = self.y - number.y

        return dx > -8 and dx < threshold and dy > 20 and dy < 40

    @property
    def number(self):
        return int(self._digits)

    @property
    def right_most_x(self):
        return max([x for x, y in self._locations])

    @property
    def left_most_x(self):
        return min([x for x, y in self._locations])

    @property
    def y(self):
        return np.mean(np.array([y for x, y in self._locations]))

    def add(self, digit_block):
        self._digits += str(digit_block.digit)
        x = digit_block.x
        y = digit_block.y
        self._locations.append((x, y))


class Composite:
    pass
