import numpy as np


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
        latex = '{}^{{{}}}'.format(self.latex, segment.latex)
        return MathSegment(region=region, latex=latex)

    def is_power_of(self, segment):
        from config import image_size

        threshold = 3 * image_size
        dx = segment.left_most_x - self.right_most_x

        dy = self.region.y - (segment.region.y + self.region.height)

        dy_min = 0
        return dx > - image_size / 2.0 and dx < threshold and dy > dy_min and dy < image_size * 4

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

    @property
    def right_most_x(self):
        return self.region.x + self.region.width

    @property
    def left_most_x(self):
        return self.region.x


class Primitive:
    def __init__(self, text, region):
        self.text = text
        self.region = region

    def is_digit(self):
        return self.text in list('0123456789')

    @property
    def x(self):
        return self.region.x

    @property
    def y(self):
        return self.region.y

    @staticmethod
    def new_primitive(digit, x, y):
        from config import image_size

        region = RectangularRegion(x - image_size / 2.0,
                                   y - image_size / 2.0, image_size, image_size)
        return Primitive(digit, region)


class RecognizedNumber(MathSegment):
    def __init__(self, region, digits=''):
        super().__init__(region, digits)

    def concatenate(self, segment):
        region = self.region.concatenate(segment)
        return RecognizedNumber(region=region, digits=self.latex + segment.latex)

    @property
    def number(self):
        return int(self.latex)

    @property
    def y(self):
        xc, yc = self.region.xy_center
        return yc


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
