import random
import os
import numpy as np
from building_blocks import RectangularRegion
from dataset_utils import examples_of_category


image_size = 45
min_size = image_size * 3


def split_interval(interval_len, n):
    a = np.random.rand(n)
    interval_fractions = a / np.sum(a)

    return [interval_len * p for p in interval_fractions]


def reduce_sections(sections, min_size):
    if len(sections) <= 1:
        return sections

    sorted_sections = sorted(sections, reverse=True)

    while np.min(sorted_sections) < min_size:
        smallest = sorted_sections.pop()
        sorted_sections[-1] += smallest
        sorted_sections.sort(reverse=True)

    return sorted_sections


def split_horizontally(width, height, max_n, min_size):
    splited_widths = split_interval(width, max_n)

    corrected_widths = reduce_sections(splited_widths, min_size)
    corrected_widths.sort()

    x = 0
    regions = []
    for w in corrected_widths:
        region = RectangularRegion(x, 0, w, height)
        x += region.width
        regions.append(region)

    return regions


def overlay_image(canvas, img, x, y):
    j, i = int(round(x)), int(round(y))
    h, w = img.shape

    if i >= canvas.shape[0] or j >= canvas.shape[1]:
        return

    if i < 0 or j < 0:
        return

    rows_available = canvas.shape[0] - i
    cols_available = canvas.shape[1] - j
    ny = min(h, rows_available)
    nx = min(w, cols_available)
    canvas[i:i + ny, j:j + nx] = np.minimum(canvas[i:i + ny, j:j + nx] + img[:ny, :nx], 255)


def visualize_image(a):
    from PIL import Image
    im = Image.frombytes('L', (a.shape[1], a.shape[0]), a.tobytes())
    im.show()


def load_random_image(csv_files_dir, cat_id):
    fname = cat_id + '.csv'

    index = random.randint(0, 1500)
    i = 0
    f = None

    for features, _ in examples_of_category(csv_files_dir, fname):
        f = features
        if i == index:
            break

        i += 1

    a = np.array(list(f), dtype=np.uint8).reshape((image_size, image_size))

    return a


def straight_line(n, slope, b, waviness):
    y = np.arange(n) * slope + b

    mask = np.random.rand() < waviness
    distorter = np.random.randint(-1, 1, n) * mask

    return y + distorter


def create_division_line_image(shape, waviness=0.5, phi=np.pi / 24):
    a = np.zeros(shape)
    h, w = shape

    k = np.tan(phi)
    b = h / 2

    margin = int(round(w * 2 / 100))

    y_indices = straight_line(w, k, b, waviness=0.5)

    for j in range(margin, w - margin):
        y = int(round(y_indices[j]))
        row = min(h - 1, max(0, y))

        a[row, j] = 255

    return a


class Synthesizer:
    def __init__(self, csv_files_dir, img_width=600, img_height=400, min_size=min_size):
        self.csv_dir = csv_files_dir
        self.img_width = img_width
        self.img_height = img_height
        self.min_size = min_size

        self.res_img = np.zeros((img_height, img_width), dtype=np.uint8)

    def synthesize_example(self):
        self.res_img = np.zeros((self.img_height, self.img_width),
                                dtype=np.uint8)

        region = RectangularRegion(0, 0, self.img_width, self.img_height)
        latex = self._fill_box(region)
        return self.res_img, latex

    def _split_canvas(self):
        n = random.randint(1, int(round(self.img_width / min_size)))
        return split_horizontally(self.img_width, self.img_height,
                                  max_n=n, min_size=self.min_size)

    def _fill_box(self, region):
        candidates = self._get_candidate_composites(region)
        composite = self._choose_composite(candidates)
        return composite.draw(region, self)

    def _choose_composite(self, candidates):
        factor = sum(map(lambda c: c.prob, candidates))

        rescaled_probs = [candidate.prob / factor for candidate in candidates]

        assert abs(sum(rescaled_probs) - 1) < 0.001
        return np.random.choice(candidates, p=rescaled_probs)

    def _get_candidate_composites(self, region):
        candidates = []
        if region.width > self.min_size:
            candidates.append(SumComposite())
            candidates.append(SubstractionComposite())
            candidates.append(ProductComposite())

        if region.height > self.min_size:
            candidates.append(DivisionComposite())

        #ndigits = self.min_size // image_size
        candidates.append(NumberComposite())

        return candidates

    def _draw_composite(self, composite, region):
        return composite.draw(region, synthesizer=self)

    def _overlay(self, img, x, y):
        overlay_image(self.res_img, img, x, y)

    def _draw_random_digit(self, x, y):
        digit = str(random.randint(0, 9))
        self._draw_random_class_image(x, y, digit)
        return digit

    def _draw_random_class_image(self, x, y, label):
        img = load_random_image(self.csv_dir, label)
        self._overlay(img, x, y)

    def _draw_div_line(self, region):
        w = int(round(region.width))

        max_slope = np.pi / 24
        phi = (random.random() - 0.5) * max_slope
        a = create_division_line_image((image_size, w), waviness=0.5, phi=phi)

        _, yc = region.xy_center
        self._overlay(a, region.x, yc - image_size / 2)


def get_drawing_strategy(csv_dir, operation):
    if operation == '+':
        return DrawSum(csv_dir)
    elif operation == '-':
        return DrawDifference(csv_dir)
    elif operation == 'times':
        return DrawProduct(csv_dir)
    elif operation == 'div':
        return DrawFraction(csv_dir)
    else:
        raise Exception('Unknown operation')


class BaseComposite:
    def draw(self, region, synthesizer):
        x, y = self._get_coordinates(region)

        op = self._get_operation()

        strategy = get_drawing_strategy(synthesizer.csv_dir, op)

        if op != 'div':
            synthesizer._draw_random_class_image(x, y, op)
        else:
            synthesizer._draw_div_line(region)

        region1, region2 = strategy.get_operand_regions(region)

        latex1 = synthesizer._fill_box(region1)
        latex2 = synthesizer._fill_box(region2)

        return strategy.combined_latex(latex1, latex2)

    def _get_coordinates(self, region):
        xc, yc = region.xy_center
        x = xc - image_size / 2
        y = yc - image_size / 2
        return x, y

    def _get_operation(self):
        raise NotImplementedError


class SumComposite(BaseComposite):
    prob = 0.2

    def _get_operation(self):
        return '+'


class SubstractionComposite(BaseComposite):
    prob = 0.2

    def _get_operation(self):
        return '-'


class ProductComposite(BaseComposite):
    prob = 0.2

    def _get_operation(self):
        return 'times'


class DivisionComposite(BaseComposite):
    prob = 0.2

    def _get_operation(self):
        return 'div'


class NumberComposite(BaseComposite):
    prob = 0.2

    def draw(self, region, synthesizer):
        x, y = self._get_coordinates(region)
        digit = str(random.randint(0, 9))
        synthesizer._draw_random_class_image(x, y, digit)
        return digit


class DrawExpression:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir

    def get_operand_regions(self, region):
        sign_size = 45 // 2
        margin = 10
        x, y = region.xy_center

        reg1 = region.left_subregion(x - sign_size - margin)
        reg2 = region.right_subregion(x + sign_size + margin)
        return reg1, reg2

    def combined_latex(self, latex1, latex2):
        raise NotImplementedError


class DrawSum(DrawExpression):
    def combined_latex(self, latex1, latex2):
        return '{} + {}'.format(latex1, latex2)


class DrawDifference(DrawExpression):
    def combined_latex(self, latex1, latex2):
        return '{} - {}'.format(latex1, latex2)


class DrawProduct(DrawExpression):
    def combined_latex(self, latex1, latex2):
        return '{} x {}'.format(latex1, latex2)


class DrawFraction(DrawExpression):
    def get_operand_regions(self, region):
        sign_size = 45 // 2
        margin = 10
        x, y = region.xy_center

        reg1 = region.subregion_above(y - sign_size - margin)
        reg2 = region.subregion_below(y + sign_size + margin)
        return reg1, reg2

    def combined_latex(self, latex1, latex2):
        return '\\\\frac{{{}}}{{{}}}'.format(latex1, latex2)


if __name__ == '__main__':
    synthesizer = Synthesizer('datasets/digits_and_operators_csv/dev')
    img, latex = synthesizer.synthesize_example()

    visualize_image(img)
    print(latex)