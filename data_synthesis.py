import random
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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

    y_indices = straight_line(w, k, b, waviness=waviness)

    for j in range(margin, w - margin):
        y = int(round(y_indices[j]))
        row = min(h - 1, max(0, y))

        a[row, j] = 255

    return a


class ImagesGenerator:
    def __init__(self, width, height, images_dir):
        self._images_dir = images_dir
        self._width = width
        self._height = height
        self._image_generators = {}
        self._make_generators()

    def _make_generators(self):
        from dataset_utils import index_to_class
        for image_class in index_to_class:
            gen = self._new_generator(image_class)
            self._image_generators[image_class] = gen

    def _new_generator(self, image_class):
        fname = '{}.csv'.format(image_class)

        while True:
            for pixels, _ in examples_of_category(self._images_dir, fname):
                a = np.array(list(pixels), dtype=np.uint8)
                yield a.reshape((self._height, self._width)), image_class

    def next_image(self, cat_class):
        gen = self._image_generators[cat_class]

        # items will appear to go in random order
        nskip = random.randint(0, 100)

        for i in range(nskip):
            next(gen)

        keras_generator = ImageDataGenerator(rotation_range=10,
                                             height_shift_range=0.05,
                                             width_shift_range=0.05,
                                             zoom_range=[0.8, 1.8],
                                             fill_mode='constant',
                                             cval=0)
        x, y = next(gen)
        x = x.reshape((1, self._height, self._width, 1))

        for x_batch, y_batch in keras_generator.flow(x, [y], batch_size=1):
            return x_batch[0].reshape((self._height, self._width))


class Canvas:
    def __init__(self, width, height, dataset_dir):
        self._images_generator = ImagesGenerator(
            image_size, image_size, dataset_dir
        )
        self._res_img = np.zeros((height, width), dtype=np.uint8)

    @property
    def image_data(self):
        return self._res_img

    def reset(self):
        self._res_img[:, :] = 0

    def _overlay(self, img, x, y):
        overlay_image(self._res_img, img, x, y)

    def draw_random_digit(self, x, y):
        digit = str(random.randint(0, 9))
        self.draw_random_class_image(x, y, digit)
        return digit

    def draw_random_class_image(self, x, y, label):
        img = self._images_generator.next_image(label)
        self._overlay(img, x, y)

    def draw_div_line(self, region):
        w = int(round(region.width))

        max_slope = np.pi / 24
        phi = (random.random() - 0.5) * max_slope
        a = create_division_line_image((image_size, w), waviness=0.5, phi=phi)

        _, yc = region.xy_center
        self._overlay(a, region.x, yc - image_size / 2)


class Synthesizer:
    def __init__(self, csv_files_dir, img_width=600, img_height=400, min_size=min_size):
        self.csv_dir = csv_files_dir
        self.img_width = img_width
        self.img_height = img_height
        self.min_size = min_size

        self.canvas = Canvas(img_width, img_height, csv_files_dir)

    def synthesize_example(self):
        self.canvas.reset()
        region = RectangularRegion(0, 0, self.img_width, self.img_height)
        latex = self._fill_box(region)
        return self.canvas.image_data, latex

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


def get_drawing_strategy(operation):
    if operation == '+':
        return DrawSum()
    elif operation == '-':
        return DrawDifference()
    elif operation == 'times':
        return DrawProduct()
    elif operation == 'div':
        return DrawFraction()
    else:
        raise Exception('Unknown operation')


class BaseComposite:
    def draw(self, region, synthesizer):
        x, y = self._get_coordinates(region)

        op = self._get_operation()

        strategy = get_drawing_strategy(op)

        if op != 'div':
            synthesizer.canvas.draw_random_class_image(x, y, op)
        else:
            synthesizer.canvas.draw_div_line(region)

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
        synthesizer.canvas.draw_random_class_image(x, y, digit)
        return digit


class DrawExpression:
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