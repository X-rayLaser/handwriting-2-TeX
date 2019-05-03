import numpy as np


class Primitive:
    def __init__(self, digit, region):
        self.digit = digit
        x, y = region.xy_center
        self.x = x
        self.y = y
        self.region = region

    def is_digit(self):
        return self.digit not in ['+', '-', 'times', 'div']

    @staticmethod
    def new_primitive(digit, x, y):
        from config import image_size

        region = RectangularRegion(x - image_size / 2,
                                   y - image_size / 2, image_size, image_size)
        return Primitive(digit, region)


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

    @property
    def region(self):
        width = abs(self.right_most_x - self.left_most_x)
        import config
        height = config.image_size
        x = self.left_most_x - config.image_size / 2.0
        y = self.y - height / 2.0

        return RectangularRegion(
            x=x, y=y,
            width=width,
            height=height
        )


class RectangularRegion:
    def __init__(self, x, y, width, height):
        from shapely.geometry import box
        self.rectbox = box(x, y, x + width, y + height)

    @property
    def xy_center(self):
        x = self.x + self.width / 2.0
        y = self.y + self.height / 2.0
        return x, y

    @property
    def x(self):
        return self.rectbox.bounds[0]

    @property
    def y(self):
        return self.rectbox.bounds[1]

    @property
    def width(self):
        x2 = self.rectbox.bounds[2]
        x1 = self.rectbox.bounds[0]
        return x2 - x1

    @property
    def height(self):
        y2 = self.rectbox.bounds[3]
        y1 = self.rectbox.bounds[1]
        return y2 - y1

    def subregion(self, x0, y0, x, y):
        from shapely.geometry import box

        new_box = box(x0, y0, x, y).intersection(self.rectbox)
        x0, y0, x, y = new_box.bounds
        width = x - x0
        height = y - y0
        return RectangularRegion(x0, y0, width, height)

    def subregion_above(self, y):
        x0 = self.x
        y0 = self.y

        x = x0 + self.width
        return self.subregion(x0, y0, x, y)

    def subregion_below(self, y):
        x0 = self.x
        x = x0 + self.width
        return self.subregion(x0, y, x, self.y + self.height)

    def left_subregion(self, x):
        return self.subregion(self.x, self.y, x, self.y + self.height)

    def right_subregion(self, x):
        xend = self.x + self.width
        return self.subregion(x, self.y, xend, self.y + self.height)

    def __contains__(self, segment):
        from shapely.geometry import box

        region = segment.region
        #b = box(region.x, region.y, region.x + region.width, region.y + region.height)

        x, y = region.xy_center
        b = box(x, y, x + 1, y + 1)
        return b.within(self.rectbox)

    def concatenate(self, segment):
        x0, y0, x, y = segment.region.rectbox.bounds
        xleft = min(x0, self.x)
        xright = max(x, self.x + self.width)

        ytop = min(y0, self.y)
        ybottom = max(y, self.y + self.height)
        return RectangularRegion(xleft, ytop, xright - xleft, ybottom - ytop)


class MathSegment:
    def __init__(self, region, latex):
        self.latex = latex
        self.region = region

    def get_difference(self, segment):
        region = self.region.concatenate(segment)
        latex = '{} - {}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def get_sum(self, segment):
        region = self.region.concatenate(segment)
        latex = '{} + {}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def get_fraction(self, segment):
        region = self.region.concatenate(segment)
        latex = '\\\\frac{{{}}}{{{}}}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def get_product(self, segment):
        region = self.region.concatenate(segment)
        latex = '{} * {}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def get_power(self, segment):
        region = self.region.concatenate(segment)
        latex = '{}^{}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def concatenate(self, segment):
        region = self.region.concatenate(segment)
        return MathSegment(region=region, latex=self.latex + segment.latex)

    def is_division_sign(self):
        return False

    def is_product_sign(self):
        return False

    def is_plus_sign(self):
        return False

    def is_minus_sign(self):
        return False


class DivisionOperator(MathSegment):
    def __init__(self, region):
        super().__init__(region, 'division')

    def is_division_sign(self):
        return True


class ProductOperator(MathSegment):
    def __init__(self, region):
        super().__init__(region, 'product')

    def is_product_sign(self):
        return True


class AdditionOperator(MathSegment):
    def __init__(self, region):
        super().__init__(region, 'addition')

    def is_plus_sign(self):
        return True


class DifferenceOperator(MathSegment):
    def __init__(self, region):
        super().__init__(region, 'difference')

    def is_minus_sign(self):
        return True
