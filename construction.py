import numpy as np
from building_blocks import RecognizedNumber, RectangularRegion, MathSegment
import config


image_size = config.image_size


class LatexBuilder:
    def generate_latex(self, segments):
        numbers = self.recognize_numbers(segments)

        if numbers:
            pows = [pow.latex for pow in self.recognize_powers(numbers)]
            return ' '.join(pows)

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

        rem = [elem for elem in list(digits) if elem.is_digit()]
        while rem:
            sorted_digits = sorted(rem, key=lambda digit: (digit.x, digit.y))

            number, rem = self.recognize_number(sorted_digits)
            numbers.append(number)
            if not rem:
                break

        return numbers

    def recognize_powers(self, numbers):
        pows = []
        numbers_in_pow = set()
        for i in range(len(numbers)):
            for j in range(len(numbers)):
                a = numbers[i]
                b = numbers[j]
                if a.is_power_of(b):
                    # todo: more robust way to calculate region

                    region = RectangularRegion(
                        x=a.left_most_x - image_size / 2.0, y=a.y - image_size / 2.0,
                        width=b.right_most_x - a.left_most_x,
                        height=image_size * 2
                    )
                    p = MathSegment(region=region, latex='{}^{{{}}}'.format(a.number, b.number))
                    pows.append(p)
                    numbers_in_pow.add(a.number)
                    numbers_in_pow.add(b.number)

        rest = [MathSegment(region=n.region, latex='{}'.format(n.number)) for n in numbers if n.number not in numbers_in_pow]

        res = pows + rest
        return res


class Reducer:
    def reduce(self, segments, region):
        res = []
        remaining_segments = list(segments)
        while remaining_segments:
            operator_segment = self.next_math_operator(remaining_segments, region)
            if operator_segment is None:
                break

            self.remove_operator(remaining_segments, operator_segment)

            first_subregion, second_subregion = self.get_subregions(operator_segment, region)
            first_segment_list = self.segments_of_region(remaining_segments, first_subregion)
            second_segment_list = self.segments_of_region(remaining_segments, second_subregion)

            first_operand = construct(first_segment_list, region=first_subregion)
            second_operand = construct(second_segment_list, region=second_subregion)

            res.append(self.apply_operation(first_operand, second_operand))

            for seg in first_segment_list:
                remaining_segments.remove(seg)

            for seg in second_segment_list:
                remaining_segments.remove(seg)

        res.extend(remaining_segments)
        return res

    def segments_of_region(self, segments, region):
        return [seg for seg in segments if seg in region]

    def next_math_operator(self, segments, region):
        raise NotImplementedError

    def get_subregions(self, operator_segment, region):
        raise NotImplementedError

    def apply_operation(self, op1, op2):
        raise NotImplementedError

    def remove_operator(self, segments, operator_segment):
        segments.remove(operator_segment)


class HorizontalReducer(Reducer):
    def any_operator_segment(self, f, segments):
        for segment in segments:
            if f(segment):
                return segment

    def get_subregions(self, operator_segment, region):
        left_one = region.left_subregion(operator_segment.region.x)
        right_one = region.right_subregion(operator_segment.region.x)
        return left_one, right_one


class FractionReducer(Reducer):
    def find_longest_division_line(self, segments):
        divlen = 0
        longest_segment = None
        for segment in segments:
            if segment.is_division_sign() and segment.region.width > divlen:
                divlen = segment.region.width
                longest_segment = segment

        return longest_segment

    def next_math_operator(self, segments, region):
        return self.find_longest_division_line(segments)

    def get_subregions(self, operator_segment, region):
        x0 = operator_segment.region.x
        x = x0 + operator_segment.region.width
        numerator_subregion = region.subregion_above(operator_segment.region.y)
        numerator_subregion = numerator_subregion.right_subregion(x0)
        numerator_subregion = numerator_subregion.left_subregion(x)

        denominator_subregion = region.subregion_below(operator_segment.region.y)
        denominator_subregion = denominator_subregion.right_subregion(x0)
        denominator_subregion = denominator_subregion.left_subregion(x)

        return numerator_subregion, denominator_subregion

    def apply_operation(self, op1, op2):
        return op1.get_fraction(op2)


class ProductReducer(HorizontalReducer):
    def next_math_operator(self, segments, region):
        return self.any_operator_segment(
            lambda seg: seg.is_product_sign(), segments
        )

    def apply_operation(self, op1, op2):
        return op1.get_product(op2)


class SumReducer(HorizontalReducer):
    def next_math_operator(self, segments, region):
        return self.any_operator_segment(
            lambda seg: seg.is_plus_sign(), segments
        )

    def apply_operation(self, op1, op2):
        return op1.get_sum(op2)


class DifferenceReducer(HorizontalReducer):
    def next_math_operator(self, segments, region):
        return self.any_operator_segment(
            lambda seg: seg.is_minus_sign(), segments
        )

    def apply_operation(self, op1, op2):
        return op1.get_difference(op2)


def get_powers(segments, region):
    builder = LatexBuilder()
    numbers = builder.recognize_numbers(segments)
    return builder.recognize_powers(numbers)


def get_fractions(segments, region):
    reducer = FractionReducer()
    return reducer.reduce(segments, region)


def get_sums(segments, region=None):
    reducer = SumReducer()
    return reducer.reduce(segments, region)


def get_differences(segments, region):
    reducer = DifferenceReducer()
    return reducer.reduce(segments, region)


def get_products(segments, region):
    reducer = ProductReducer()
    return reducer.reduce(segments, region)


def construct_latex(segments, width, height):
    region = RectangularRegion(0, 0, width, height)

    # todo: move this line back to construct function
    power_segments = get_powers(segments, region)

    sign_segments = []

    from building_blocks import DivisionOperator, AdditionOperator, DifferenceOperator, ProductOperator
    for seg in segments:
        if not seg.is_digit():
            width = image_size
            height = image_size
            if seg.digit == '+':
                cls_name = AdditionOperator
            elif seg.digit == '-':
                cls_name = DifferenceOperator
            elif seg.digit == 'times':
                cls_name = ProductOperator
            elif seg.digit == 'div':
                cls_name = DivisionOperator
                height = 0
                width = seg.region.width
            else:
                raise Exception('WOooowoow! digit is {}'.format(seg.digit))

            top_left_x = seg.x - width / 2.0
            top_left_y = seg.y - height / 2.0
            sign_segments.append(cls_name(
                region=RectangularRegion(top_left_x, top_left_y, width, height)
            ))

    result = construct(power_segments + sign_segments, region)
    return result.latex


def construct(segments, region):
    # todo: make line below work
    #segments = get_powers(segments, region)
    segments = get_fractions(segments, region)
    segments = get_sums(segments, region)
    segments = get_differences(segments, region)
    segments = get_products(segments, region)

    if segments:
        return segments[0]

    return MathSegment(region, '?')
