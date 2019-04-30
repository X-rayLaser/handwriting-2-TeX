import random
import os
import numpy as np
from building_blocks import RectangularRegion
from dataset_utils import examples_of_category


image_size = 45


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

    from PIL import Image
    a = np.array(list(f), dtype=np.uint8).reshape((image_size, image_size))

    im = Image.frombytes('L', (image_size, image_size), a.tobytes())
    #im.show()
    return a


class Synthesizer:
    def __init__(self, csv_files_dir, img_width=700, img_height=650, min_size=110):
        self.csv_dir = csv_files_dir
        self.img_width = img_width
        self.img_height = img_height
        self.min_size = min_size

        self.res_img = np.zeros((img_height, img_width), dtype=np.uint8)

    def synthesize_example(self):
        self.res_img = np.zeros((self.img_height, self.img_width),
                                dtype=np.uint8)

        regions = self._split_canvas()

        latex = ''

        for i in range(len(regions)):
            region = regions[i]
            if i % 2 == 0:
                latex += self._fill_box(region)
            else:
                op = random.choice(['+', '-', 'times'])
                self._draw_random_class_image(region, op)

                strategy = get_drawing_strategy(self.csv_dir, op)
                latex = strategy.combined_latex(latex, ' ')

        visualize_image(self.res_img)
        print(latex)
        return latex

    def _split_canvas(self):
        width, height, min_size = self.img_width, self.img_height, self.min_size

        n = random.randint(1, int(round(width / min_size)))
        return split_horizontally(width, height, max_n=n, min_size=min_size)

    def _fill_box(self, region):
        min_size = self.min_size
        if region.width <= min_size and region.height <= min_size:
            digit = str(random.randint(0, 9))
            return self._draw_random_class_image(region, digit)

        op = self._choose_operation(region)

        strategy = get_drawing_strategy(self.csv_dir, op)

        if op != 'div':
            self._draw_random_class_image(region, op)
        else:
            self._draw_div_line(region)

        region1, region2 = strategy.get_operand_regions(region)

        latex1 = self._fill_box(region1)
        latex2 = self._fill_box(region2)

        return strategy.combined_latex(latex1, latex2)

    def _choose_operation(self, region):
        min_size = self.min_size

        if region.width > min_size and region.height > min_size:
            operation_options = ['+', '-', 'times', 'div']
        elif region.width > min_size:
            operation_options = ['+', '-', 'times']
        elif region.height > min_size:
            operation_options = ['div']
        else:
            raise Exception('Oops')

        return random.choice(operation_options)

    def _overlay(self, region, img):
        x, y = region.xy_center
        j = int(round(x - 45 // 2))
        i = int(round(y - 45 // 2))

        h, w = img.shape
        try:
            self.res_img[i:i + h, j:j + w] = img[:, :]
        except:
            pass

    def _draw_random_class_image(self, region, label):
        img = load_random_image(self.csv_dir, label)
        self._overlay(region, img)

    def _draw_div_line(self, region):
        w = int(round(region.width))
        a = np.zeros((5, w))

        a[1:3, 5:-5] = 220
        self._overlay(region, a)


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


class DrawExpression:
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir

    def get_operand_regions(self, region):
        sign_size = 45 // 2
        x, y = region.xy_center

        reg1 = region.left_subregion(x - sign_size)
        reg2 = region.right_subregion(x + sign_size)
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
        x, y = region.xy_center

        reg1 = region.subregion_above(y - sign_size)
        reg2 = region.subregion_below(y + sign_size)
        return reg1, reg2

    def combined_latex(self, latex1, latex2):
        return '\\\\frac{}{}'.format(latex1, latex2)


if __name__ == '__main__':
    synthesizer = Synthesizer('datasets/digits_and_operators_csv/dev')
    synthesizer.synthesize_example()
