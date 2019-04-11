import numpy as np


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

    def add(self, digit, x, y):
        self._digits += digit
        self._locations.append((x, y))


class LatexBuilder:
    def generate_latex(self, segments):
        numbers = self.recognize_numbers(segments)

        if numbers:
            return self.recognize_powers(numbers)

    def _nearest_neighbor(self, digits, x, y):
        filtered = []
        remaining = []
        for digit, row, col in digits:
            if abs(self._phi(x, y, col, row)) < np.pi / 8:
                filtered.append((digit, row, col))
            else:
                remaining.append((digit, row, col))

        def distance(triple):
            digit, row, col = triple
            return self._distance(x, y, col, row)

        sorted_digits = sorted(filtered, key=distance, reverse=True)
        if not sorted_digits:
            return

        first_digit, row, col = sorted_digits.pop()
        if self._are_neighbors(x, y, col, row):
            return first_digit, row, col

    def _are_neighbors(self, x1, y1, x2, y2):
        d = self._distance(x1, y1, x2, y2)
        phi = self._phi(x1, y1, x2, y2)

        return d < 50 and abs(phi) < np.pi / 16

    def _distance(self, x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def _phi(self, x1, y1, x2, y2):
        dy = y2 - y1
        dx = x2 - x1
        epsilon = 10 ** (-8)
        return np.arctan(dy / (dx + epsilon))

    def recognize_number(self, digits):
        remaining = list(digits)
        remaining.reverse()
        digit, row, col = remaining.pop()

        current_number = RecognizedNumber()
        current_number.add(str(digit), col, row)
        while remaining:
            res = self._nearest_neighbor(remaining, col, row)

            if res is None:
                return current_number, remaining

            neighbor, row, col = res
            remaining.remove((neighbor, row, col))
            current_number.add(str(neighbor), col, row)

        return current_number, remaining

    def recognize_numbers(self, digits):
        numbers = []

        rem = list(digits)
        while True:
            sorted_digits = sorted(rem, key=lambda t: (t[2], t[1]))

            number, rem = self.recognize_number(sorted_digits)
            numbers.append(number)
            if not rem:
                return numbers

    def recognize_powers(self, numbers):
        pows = []
        numbers_in_pow = set()
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                a = numbers[i]
                b = numbers[j]
                if a.is_power_of(b):
                    p = '{}^{{{}}}'.format(a.number, b.number)
                    pows.append(p)
                    numbers_in_pow.add(a.number)
                    numbers_in_pow.add(b.number)

        rest = [str(n.number) for n in numbers if n.number not in numbers_in_pow]

        res = pows + rest
        return ' '.join(res)