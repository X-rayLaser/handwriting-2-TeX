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

        self._volume = np.zeros((grid_size, grid_size, self.depth), dtype=np.uint8)

        self._boxes = []

    @staticmethod
    def unroll_classes(flat_volume, grid_size, num_classes):
        confidences_score_size = 1
        box_size = 4
        label_size = 1
        depth = confidences_score_size + 4 + label_size

        array_3d = flat_volume.reshape((grid_size, grid_size, depth))
        h, w, d = array_3d.shape

        new_depth = confidences_score_size + box_size + num_classes
        res = np.zeros((w, h, new_depth))

        for i in range(h):
            for j in range(w):
                v = array_3d[i, j, :]
                label_index = confidences_score_size + box_size
                label = v[label_index]

                from keras.utils import to_categorical

                class_distribution = to_categorical(label,
                                                    num_classes=num_classes)
                res[i, j, :] = np.concatenate((array_3d[i, j, :label_index],
                                               class_distribution))

        return res

    def detects_collision(self, bounding_box):
        comparison_box = self._to_shapely_box(bounding_box)

        for box in self._boxes:
            b = self._to_shapely_box(box)
            if comparison_box.intersection(b).area > 0:
                return True

        return False

    def _to_shapely_box(self, bounding_box):
        from shapely.geometry import box

        xc, yc, width, height = bounding_box
        x = xc - width // 2
        y = yc - height // 2
        return box(x, y, x + width, y + height)

    def add_item(self, bounding_box, class_index):
        xc, yc, width, height = bounding_box
        col, row = self.position_to_grid_cell(xc, yc)

        detection_confidence = 1

        #box_vector = self.get_bounding_box(xc, yc, col, row, width=width,
        #                                   height=height)

        box_vector = [xc, yc, width, height]
        self._boxes.append(bounding_box)

        self._volume[row, col] = self.make_prediction_vector(
            detection_confidence, box_vector, class_index
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

    def make_prediction_vector(self, confidence, bounding_box, class_index):
        return np.concatenate(([confidence], bounding_box, [class_index]))

    @property
    def output_volume(self):
        return self._volume

    @property
    def depth(self):
        confidence_score = 1
        bounding_box_size = 4
        label_size = 1
        return confidence_score + bounding_box_size + label_size

    @property
    def cell_width(self):
        return self.img_width / self.grid_size

    @property
    def cell_height(self):
        return self.img_height / self.grid_size


class BoundingBoxSuggestionError(Exception):
    pass


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

    def try_choosing_position(self, volume, max_retries=10):
        for i in range(max_retries):
            xc, yc = self.choose_global_position()

            width, height = 45, 45
            bounding_box = (xc, yc, width, height)

            if not volume.detects_collision(bounding_box):
                return bounding_box

        raise BoundingBoxSuggestionError()

    def make_example(self, elements=40):
        canvas = Canvas(self.img_width, self.img_height, self.primitives_dir)

        volume = YoloVolume(self.img_width, self.img_height, self.grid_size, self.num_classes)

        for _ in range(elements):
            class_index, category = self.choose_category()

            try:
                bounding_box = self.try_choosing_position(volume)
            except BoundingBoxSuggestionError:
                continue
            else:
                volume.add_item(bounding_box, class_index=class_index)

                xc, yc, _, _ = bounding_box
                x = xc - 45 // 2
                y = yc - 45 // 2
                canvas.draw_random_class_image(x, y, category)

        input = canvas.image_data
        return input, volume.output_volume

    def generate_dataset(self, num_examples=5):

        for i in range(num_examples):
            elements = random.randint(0, 120)
            input, output = self.make_example(elements=elements)
            print(output)


class YoloDatasetHome:
    def __init__(self, dataset_root):
        self._dataset_root = dataset_root
        os.makedirs(self.train_path, exist_ok=True)
        os.makedirs(self.dev_path, exist_ok=True)
        os.makedirs(self.test_path, exist_ok=True)

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
        n = get_input_size(self.config)

        for row in self.all_rows:
            xy_list = map(np.uint8, row)
            x = xy_list[:n]
            y = xy_list[n:]
            yield x, y

    def flow_with_preload(self, dataset_path, mini_batch_size, normalize=False):
        preloader = Preloader(dataset_path, self.config)
        inputs, outputs = preloader.preload()

        m = len(inputs)

        grid_size = self.config['output_config']['grid_size']
        num_classes = self.config['output_config']['num_classes']

        while True:
            # todo: shuffling analog (random picking) for without preload case
            #x, labels = shuffle_data(x, labels)

            for i in range(0, outputs.shape[0], mini_batch_size):
                x_batch = inputs[i:i + mini_batch_size, :]
                input_shape = (x_batch.shape[0], ) + tuple(self.config['input_config']['shape'])
                x_batch = x_batch.reshape(input_shape)

                if normalize:
                    x_batch = x_batch / 255.0

                y_batch = outputs[i:i + mini_batch_size, :]

                y_batch = [YoloVolume.unroll_classes(y, grid_size, num_classes)
                           for y in y_batch]

                yield x_batch, np.array(y_batch)

    def flow(self, dataset_path, mini_batch_size, normalize=False):
        while True:
            pass

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
            'output_config': output_config,
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


def total_size(shape):
    n = 1

    for dim_size in shape:
        n *= dim_size

    return n


def get_input_size(config):
    input_shape = config['input_config']['shape']
    return total_size(input_shape)


def get_output_size(config):
    output_shape = config['output_config']['shape']
    return total_size(output_shape)


def generate_dataset(primitives_source, destination_dir, num_examples):
    image_width = 350
    image_height = 350
    grid_size = 9
    num_classes = 14

    input_config = {
        'image_width': image_width,
        'image_height': image_height,
        'shape': (image_width, image_height)
    }

    conf_score_size = 1
    box_size = 4
    label_size = 1

    depth = conf_score_size + box_size + label_size

    output_config = {
        'grid_size': grid_size,
        'num_classes': num_classes,
        'shape': (grid_size, grid_size, depth)
    }

    yolo_home = YoloDatasetHome.initialize_dataset(destination_dir,
                                                   num_examples=num_examples,
                                                   input_config=input_config,
                                                   output_config=output_config)

    gen = YoloDataGenerator(image_width, image_height, primitives_source,
                            grid_size=grid_size, num_classes=num_classes)

    for i in range(num_examples):
        n = 30
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
    path, num_examples, input_size, output_size = args

    dtype = np.uint8
    x_batch = np.zeros((num_examples, input_size), dtype=dtype)
    y_batch = np.zeros((num_examples, output_size), dtype=dtype)

    i = 0
    with open(path, 'r', newline='') as f:
        for row in csv.reader(f):
            xy = list(map(np.uint8, row))
            x_batch[i, :] = xy[:input_size]
            y_batch[i, :] = xy[input_size:]
            i += 1

    return x_batch, y_batch


class Preloader:
    def __init__(self, split_dir_path, dataset_config):
        self._path = split_dir_path
        self._config = dataset_config

    def preload(self, max_workers=4):
        m, _ = self._number_of_examples()
        x = np.zeros((m, self._input_size), dtype=np.uint8)
        y = np.zeros((m, self._output_size), dtype=np.uint8)

        results = self._parallel_load(max_workers)

        i = 0

        for x_batch, y_batch in results:
            m = x_batch.shape[0]
            x[i:i + m, :] = x_batch
            y[i:i + m] = y_batch
            i += m

        return x, y

    def _parallel_load(self, max_workers):
        _, counts = self._number_of_examples()

        arg_list = []

        for fname in os.listdir(self._path):
            file_path = os.path.join(self._path, fname)
            num_examples = counts[fname]
            arg_list.append((file_path, num_examples, self._input_size, self._output_size))

        pool = ProcessPoolExecutor(max_workers=max_workers)
        return list(pool.map(load_part, arg_list))

    @property
    def _input_size(self):
        return get_input_size(self._config)

    @property
    def _output_size(self):
        return get_output_size(self._config)

    def _number_of_examples(self):
        counts = {}
        for fname in os.listdir(self._path):
            file_path = os.path.join(self._path, fname)
            counts[fname] = 0
            with open(file_path, 'r', newline='') as f:
                for _ in csv.reader(f):
                    counts[fname] += 1

        total_count = sum(counts.values())
        return total_count, counts


if __name__ == '__main__':
    destination_dir = '../datasets/yolo_dataset'
    csv_dir = '../datasets/digits_and_operators_csv/train'
    generate_dataset(primitives_source=csv_dir, destination_dir=destination_dir, num_examples=5)

    train_dir = os.path.join(destination_dir, 'train')

    yolo_home = YoloDatasetHome(destination_dir)
    counter = 0
    for x_batch, y_batch in yolo_home.flow_with_preload(train_dir, mini_batch_size=2):
        x = x_batch[0]
        y = y_batch[0]
        visualize_image(x)

        counter += 1
        print(y_batch.shape)
        if counter > 2:
            break
