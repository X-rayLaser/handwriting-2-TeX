import numpy as np
from building_blocks import RecognizedNumber


class LatexBuilder:
    def generate_latex(self, segments):
        numbers = self.recognize_numbers(segments)

        if numbers:
            return self.recognize_powers(numbers)

    def _nearest_neighbor(self, digits, current_digit):
        x = current_digit.x
        y = current_digit.y
        filtered = []
        remaining = []
        for digit_segment in digits:
            if abs(self._phi(x, y, digit_segment.x, digit_segment.y)) < np.pi / 8:
                filtered.append(digit_segment)
            else:
                remaining.append(digit_segment)

        def distance(digit_segment):
            return self._distance(x, y, digit_segment.x, digit_segment.y)

        sorted_digits = sorted(filtered, key=distance, reverse=True)
        if not sorted_digits:
            return

        first_digit = sorted_digits.pop()
        if self._are_neighbors(x, y, first_digit.x, first_digit.y):
            return first_digit

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
        digit = remaining.pop()

        current_number = RecognizedNumber()
        current_number.add(digit)
        while remaining:
            digit = self._nearest_neighbor(remaining, digit)

            if digit is None:
                return current_number, remaining

            remaining.remove(digit)
            current_number.add(digit)

        return current_number, remaining

    def recognize_numbers(self, digits):
        numbers = []

        rem = list(digits)
        while True:
            sorted_digits = sorted(rem, key=lambda digit: (digit.x, digit.y))

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