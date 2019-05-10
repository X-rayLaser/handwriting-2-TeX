import random
import csv
import os
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from data_synthesis import Canvas, visualize_image


class YoloVolume:
    def __init__(self, image_width, image_height, grid_size, num_classes):
        self.img_width = image_width
        self.img_height = image_height
        self.grid_size = grid_size
        self.num_classes = num_classes

        self._volume = np.zeros((grid_size, grid_size, self.depth))

    def add_item(self, xy_center, width, height, class_index):
        xc, yc = xy_center

        col, row = self.position_to_grid_cell(xc, yc)

        detection_confidence = 1

        box_vector = self.get_bounding_box(xc, yc, col, row, width=width,
                                           height=height)

        from keras.utils import to_categorical

        one_hot_classes = to_categorical(class_index, num_classes=self.num_classes)

        self._volume[row, col] = self.make_prediction_vector(
            detection_confidence, box_vector, one_hot_classes
        )

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

    def make_prediction_vector(self, confidence, bounding_box, class_distribution):
        return np.concatenate(([confidence], bounding_box, class_distribution))

    @property
    def output_volume(self):
        return self._volume

    @property
    def depth(self):
        confidence_score = 1
        bounding_box_size = 4
        return confidence_score + bounding_box_size + self.num_classes

    @property
    def cell_width(self):
        return self.img_width / self.grid_size

    @property
    def cell_height(self):
        return self.img_height / self.grid_size


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

    def make_prediction_vector(self, confidence, bounding_box, class_index):
        return np.concatenate(([confidence], bounding_box, [class_index]))

    def make_example(self, elements=40):
        canvas = Canvas(self.img_width, self.img_height, self.primitives_dir)

        volume = YoloVolume(self.img_width, self.img_height, self.grid_size, self.num_classes)

        for _ in range(elements):
            class_index, category = self.choose_category()
            xc, yc = self.choose_global_position()

            volume.add_item((xc, yc), width=45, height=45, class_index=class_index)

            x = xc - 45 / 2
            y = yc - 45 / 2
            canvas.draw_random_class_image(x, y, category)

        input = canvas.image_data
        return input, volume.output_volume

    def generate_dataset(self, num_examples=5):

        for i in range(num_examples):
            elements = random.randint(0, 120)
            input, output = self.make_example(elements=elements)

            visualize_image(input)
            print(output)


class YoloDatasetHome:
    def __init__(self, dataset_root):
        self._dataset_root = dataset_root
        os.makedirs(self.train_path)
        os.makedirs(self.dev_path)
        os.makedirs(self.test_path)

    @staticmethod
    def initialize_dataset(dataset_root, num_examples,
                           input_config, output_config,
                           training_fraction=0.7, number_of_parts=4):

        os.makedirs(dataset_root, exist_ok=True)

        conf_path = os.path.join(dataset_root, 'config.csv')

        YoloDatasetHome.create_configuration(conf_path,
                                             num_examples,
                                             input_config,
                                             output_config,
                                             training_fraction,
                                             number_of_parts)

        return YoloDatasetHome(dataset_root)

    @property
    def config_path(self):
        return os.path.join(self._dataset_root, 'config.csv')

    @property
    def all_examples_path(self):
        return ''

    @property
    def config(self):
        with open(self.config_path, 'r') as f:
            s = f.read()

        d = json.loads(s)
        return d

    @property
    def root_path(self):
        return self._dataset_root

    @property
    def train_path(self):
        return os.path.join(self.root_path, 'train')

    @property
    def dev_path(self):
        return os.path.join(self.root_path, 'dev')

    @property
    def test_path(self):
        return os.path.join(self.root_path, 'test')

    def get_all_examples(self):
        input_shape = self.config['input_config']['shape']

        n = 1

        for dim_size in input_shape:
            n *= dim_size

        for row in self.all_rows:
            xy_list = map(np.uint8, row)
            x = xy_list[:n]
            y = xy_list[n:]
            yield x, y

    @property
    def all_rows(self):
        split_roots = [self.train_path, self.dev_path, self.test_path]

        for split_path in split_roots:
            for fname in os.listdir(split_path):
                path = os.path.join(split_path, fname)

                with open(path, 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=',')
                    for row in reader:
                        yield row

    @staticmethod
    def create_configuration(path, num_examples, input_config, output_config,
                             training_fraction, number_of_parts):
        d = {
            'num_examples': num_examples,
            'input_config': input_config,
            'output_config': output_config
            'training_fraction': training_fraction,
            'number_of_parts': number_of_parts
        }

        s = json.dumps(d)

        with open(path, 'w') as f:
            f.write(s)

    def add_example(self, input, output):
        train_fraction = self.config['training_fraction']
        parts = self.config['number_of_parts']

        destination = self.random_destination(train_fraction, parts)

        example = input.flatten().tolist() + output.flatten().tolist()
        with open(destination, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(example)

        self._flush_config()

    def random_destination(self, train_fraction, parts):
        split_dir = self._choose_split_directory(train_fraction)
        part_file = self._choose_part_file(parts)
        return os.path.join(self._dataset_root, split_dir, part_file)

    def _choose_split_directory(self, train_fraction):
        remaining_fraction = 1 - train_fraction
        dev_test_fraction = remaining_fraction / 2.0
        split_pmf = [train_fraction, dev_test_fraction, dev_test_fraction]

        split_names = ['train', 'dev', 'test']

        return np.random.choice(split_names, p=split_pmf)

    def _choose_part_file(self, parts):
        part_index = random.randint(1, parts)
        return 'part{}.csv'.format(part_index)

    def _flush_config(self):
        s = json.dumps(self.config)
        with open(self.config_path, 'w') as f:
            f.write(s)


def generate_dataset(primitives_source, destination_dir, num_examples):
    image_width = 350
    image_height = 350
    grid_size = 9
    num_classes = 14

    input_config = {
        'image_width': image_width,
        'image_height': image_height,
    }

    output_config = {
        'grid_size': grid_size,
        'num_classes': num_classes
    }

    yolo_home = YoloDatasetHome.initialize_dataset(destination_dir,
                                                   num_examples=num_examples,
                                                   input_config=input_config,
                                                   output_config=output_config)

    gen = YoloDataGenerator(image_width, image_height, primitives_source,
                            grid_size=grid_size, num_classes=num_classes)

    for i in range(num_examples):
        n = 4
        input, output = gen.make_example(elements=n)
        yolo_home.add_example(input, output)


def precompute_features(dataset_root, destination):
    yolo_source = YoloDatasetHome(dataset_root)

    m = yolo_source.config['num_examples']
    input_config = yolo_source.config['input_config']
    output_config = yolo_source.config['output_config']
    training_fraction = yolo_source.config['training_fraction']
    number_of_parts = yolo_source.config['number_of_parts']

    yolo_destination = YoloDatasetHome.initialize_dataset(destination,
                                                          num_examples=m,
                                                          input_config=input_config,
                                                          output_config=output_config,
                                                          training_fraction=training_fraction,
                                                          number_of_parts=number_of_parts)

    for x, y in yolo_source.get_all_examples():
        #preprocess

        example = np.array([])
        input, output = example
        yolo_destination.add_example(input, output)


def load_part(args):
    path, num_examples, options = args
    # todo: finish this one


class Preloader:
    def __init__(self, split_dir_path, dataset_config):
        self._path = split_dir_path
        self._config = dataset_config

    def preload(self, max_workers=4):
        pool = ProcessPoolExecutor(max_workers=max_workers)

        arg_list = [(csv_files_dir, fname, category_counts)
                    for fname in os.listdir(self._path)]

        results = list(pool.map(all_examples_with_category, arg_list))
        # todo: finish impl

    def _number_of_examples(self):
        counts = {}
        for fname in os.listdir(self._path):
            file_path = os.path.join(self._path, fname)
            counts[fname] = 0
            with open(file_path, 'r', newline='') as f:
                for _ in csv.reader(f):
                    counts[fname] += 1

        return counts


def load_pretrained_examples(path, num_examples, width, height, channels, grid_size, dtype=np.uint8):
    m = num_examples

    confidence_score_size = 1
    bounding_box_size = 4
    label_size = 1
    ydepth = confidence_score_size + bounding_box_size + label_size

    x = np.zeros((m, width * height * channels), dtype=dtype)
    y = np.zeros((m, grid_size * grid_size * ydepth), dtype=np.uint8)

    with open(path, 'r', newline='') as fin:
        reader = csv.reader(fin, delimiter=',')

        i = 0
        for row in reader:
            line = list(map(dtype, row))
            separator = width * height * channels
            x[i, :] = line[:separator]
            y[i, :] = line[separator:]
            i += 1

    return x, y


def preload_raw_data(dataset_dir, width, height, grid_size):
    counter_file_name = 'counter.json'
    counter_path = os.path.join(dataset_dir, counter_file_name)

    import json

    with open(counter_path, 'r') as f:
        s = f.read()
        data_sizes = json.loads(s)

    m = sum(data_sizes.values())

    x = np.zeros((m, width, height))

    csv_files = [fname for fname in os.listdir(dataset_dir)]

    load_examples(path, num_examples, width, height, channels, grid_size, dtype=np.uint8)


def create_dataset(destination_dir):
    csv_dir = '../datasets/digits_and_operators_csv/train'
    generate_dataset(primitives_source=csv_dir, destination_dir=destination_dir, num_examples=2)
    precompute_features(destination_dir)
    split_data(destination_dir)


if __name__ == '__main__':
    destination_dir = '../datasets/yolo_dataset'
    create_dataset(destination_dir)
    preload_data(destination_dir)
    #csv_dir = '../datasets/digits_and_operators_csv/dev'

    #gen = YoloDataGenerator(350, 250, csv_dir)
    #gen.generate_dataset(2)
