import numpy as np
from building_blocks import RecognizedNumber, RectangularRegion, MathSegment
import config


image_size = config.image_size


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
        hor_padding = image_size // 2
        x0 = operator_segment.region.x
        x = x0 + operator_segment.region.width
        numerator_subregion = region.subregion_above(operator_segment.region.y)
        numerator_subregion = numerator_subregion.right_subregion(x0 - hor_padding)
        numerator_subregion = numerator_subregion.left_subregion(x + hor_padding)

        denominator_subregion = region.subregion_below(operator_segment.region.y)
        denominator_subregion = denominator_subregion.right_subregion(x0 - hor_padding)
        denominator_subregion = denominator_subregion.left_subregion(x + hor_padding)

        return numerator_subregion, denominator_subregion

    def apply_operation(self, op1, op2):
        return op1.get_fraction(op2)


class PowerReducer(Reducer):
    def next_math_operator(self, segments, region):
        for i in range(len(segments)):
            for j in range(len(segments)):
                a = segments[i]
                b = segments[j]
                if a.is_power_of(b):
                    dx = b.left_most_x - a.right_most_x
                    dy = a.region.y - (b.region.y + b.region.height)
                    x = a.right_most_x + dx / 2.0
                    y = a.y - dy / 2.0
                    point_region = RectangularRegion(x, y, 0, 0)
                    dummy = MathSegment(point_region, latex='')
                    return dummy

    def get_subregions(self, operator_segment, region):
        x = operator_segment.region.x
        y = operator_segment.region.y
        first = region.left_subregion(x)
        second = region.right_subregion(x)
        return first, second

    def apply_operation(self, op1, op2):
        return op1.get_power(op2)

    def remove_operator(self, segments, operator_segment):
        pass


def discriminate_primitives(primitives):
    sign_segments = []
    digits = []

    from building_blocks import DivisionOperator, AdditionOperator, DifferenceOperator, ProductOperator
    for seg in primitives:
        if not seg.is_digit():
            width = image_size
            height = image_size
            if seg.text == '+':
                cls_name = AdditionOperator
            elif seg.text == '-':
                cls_name = DifferenceOperator
            elif seg.text == 'times':
                cls_name = ProductOperator
            elif seg.text == 'div':
                cls_name = DivisionOperator
                height = 0
                width = seg.region.width
            else:
                raise Exception('WOooowoow! digit is {}'.format(seg.text))

            top_left_x = seg.x
            top_left_y = seg.y
            sign_segments.append(cls_name(
                region=RectangularRegion(top_left_x, top_left_y, width, height)
            ))
        else:
            digits.append(RecognizedNumber(seg.region, seg.text))

    return sign_segments, digits


def construct_latex(segments, width, height):
    region = RectangularRegion(0, 0, width, height)

    sign_segments, digits = discriminate_primitives(segments)
    all_segments = digits + sign_segments
    return construct(all_segments, region).latex


def construct(segments, region):
    segments = get_fractions(segments, region)
    return reduce_terms(segments, region)


def get_powers(numbers, region):
    reducer = PowerReducer()
    return reducer.reduce(numbers, region)


def get_fractions(segments, region):
    reducer = FractionReducer()
    return reducer.reduce(segments, region)


def reduce_terms(segments, region):
    if not segments:
        return MathSegment(region, '?')

    def sortf(segment):
        x, y = segment.region.xy_center
        return x, y

    left_to_right = sorted(segments, key=sortf)

    from building_blocks import AdditionOperator, DifferenceOperator, ProductOperator

    def is_sign(segment):
        return isinstance(segment, (AdditionOperator, DifferenceOperator, ProductOperator))

    sign_segments = list(filter(is_sign, left_to_right))

    if len(sign_segments) == 0:
        return construct_power(left_to_right, region)

    operator_segment = sign_segments[0]

    left_subregion = region.left_subregion(operator_segment.region.x)
    right_subregion = region.right_subregion(
        operator_segment.region.x + operator_segment.region.width
    )

    left_subregion_segments = [seg for seg in left_to_right if seg in left_subregion]
    right_subregion_segments = [seg for seg in left_to_right if seg in right_subregion]

    first_operand = construct_power(left_subregion_segments, region=left_subregion)
    second_operand = reduce_terms(right_subregion_segments, region=right_subregion)

    if operator_segment.is_product_sign():
        return first_operand.get_product(second_operand)
    elif operator_segment.is_plus_sign():
        return first_operand.get_sum(second_operand)
    elif operator_segment.is_minus_sign():
        return first_operand.get_difference(second_operand)
    else:
        raise Exception('Unknown sign ' + operator_segment.latex)


def construct_numbers(segments, region):
    def f(digit):
        x, y = digit.region.xy_center
        return x, y

    sorted_by_xy = sorted(segments, key=f, reverse=True)

    if len(sorted_by_xy) == 0:
        return []

    previous_digit = sorted_by_xy.pop()
    numbers = [
        previous_digit
    ]

    while len(sorted_by_xy) > 0:
        single_digit_number = sorted_by_xy.pop()

        x1, y1 = previous_digit.region.xy_center

        x2, y2 = single_digit_number.region.xy_center

        if are_neighbors(x1, y1, x2, y2):
            current_number = numbers[-1]
            numbers[-1] = current_number.concatenate(single_digit_number)
        else:
            numbers.append(single_digit_number)

        previous_digit = single_digit_number

    return numbers


def construct_power(segments, region):
    if not segments:
        return MathSegment(region, '?')

    numbers = construct_numbers(segments, region)

    if len(numbers) == 1:
        return numbers[0]

    if not numbers:
        return MathSegment(region, '?')

    if len(numbers) > 2:
        return numbers[0]

    a = numbers[0]
    b = numbers[1]
    if a.is_power_of(b):
        return a.get_power(b)
    elif b.is_power_of(b):
        return b.get_power(b)
    else:
        return numbers[0]


def are_neighbors(x1, y1, x2, y2):
    d = euclidean_distance(x1, y1, x2, y2)
    phi = angle(x1, y1, x2, y2)

    return d < 10 * image_size and abs(phi) < np.pi / 16


def euclidean_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def angle(x1, y1, x2, y2):
    dy = y2 - y1
    dx = x2 - x1
    epsilon = 10 ** (-8)
    return np.arctan(dy / (dx + epsilon))
