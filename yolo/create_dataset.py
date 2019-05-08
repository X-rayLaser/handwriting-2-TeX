import random
import numpy as np
from data_synthesis import Canvas, visualize_image


class YoloDataGenerator:
    def __init__(self, img_width, img_height, primitives_dir, grid_size=9, num_classes=14):
        self.img_width = img_width
        self.img_height = img_height
        self.primitives_dir = primitives_dir
        self.grid_size = grid_size
        self.num_classes = num_classes

    @property
    def cell_width(self):
        return self.img_width / self.grid_size

    @property
    def cell_height(self):
        return self.img_height / self.grid_size

    def choose_category(self):
        from dataset_utils import index_to_class, class_to_index

        c = random.choice(index_to_class)
        index = class_to_index[c]
        return index, c

    def choose_global_position(self):
        x = random.randint(0, self.img_width - 1)
        y = random.randint(0, self.img_height - 1)

        return x, y

    def position_to_grid_cell(self, xc, yc):
        x = xc // self.cell_width
        y = yc // self.cell_height

        return int(x), int(y)

    def relative_position(self, global_position, cell, cell_size):
        cell_position = cell_size * cell
        return (global_position - cell_position) / cell_size

    def get_bounding_box(self, xc, yc, col, row, width, height):
        j = col
        i = row

        xrel = self.relative_position(xc, j, self.cell_width)
        yrel = self.relative_position(yc, i, self.cell_height)

        w = width / self.cell_width
        h = height / self.cell_height

        return np.array([xrel, yrel, w, h])

    def make_prediction_vector(self, confidence, bounding_box, class_index):
        return np.concatenate(([confidence], bounding_box, [class_index]))

    def make_example(self, elements=40):
        canvas = Canvas(self.img_width, self.img_height, self.primitives_dir)

        confidence_size = 1
        bounding_box_size = 4
        labels_size = 1

        prediction_vector_size = confidence_size + bounding_box_size + labels_size
        output = np.zeros((self.grid_size,
                           self.grid_size,
                           prediction_vector_size))

        for _ in range(elements):
            xc, yc = self.choose_global_position()

            col, row = self.position_to_grid_cell(xc, yc)

            detection_confidence = 1

            box_vector = self.get_bounding_box(xc, yc, col, row, width=45,
                                               height=45)

            class_index, category = self.choose_category()

            x = xc - 45 / 2
            y = yc - 45 / 2
            canvas.draw_random_class_image(x, y, category)

            output[row, col] = self.make_prediction_vector(detection_confidence,
                                                           box_vector, class_index)

        input = canvas.image_data
        return input, output

    def generate_dataset(self, num_examples=5):

        for i in range(num_examples):
            elements = random.randint(0, 120)
            input, output = self.make_example(elements=elements)

            visualize_image(input)
            print(output)


if __name__ == '__main__':
    csv_dir = '../datasets/digits_and_operators_csv/dev'

    gen = YoloDataGenerator(350, 250, csv_dir)
    gen.generate_dataset(2)
