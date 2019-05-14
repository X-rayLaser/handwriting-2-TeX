import random
import csv
import os
import json
import numpy as np
from .yolo_utils import get_input_size
from .preload import Preloader
from .volume import YoloVolume


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
            xy_list = list(map(np.uint8, row))
            x = xy_list[:n]
            y = xy_list[n:]
            yield x, y

    def flow_with_preload(self, dataset_path, mini_batch_size=32, normalize=False):
        preloader = Preloader(dataset_path, self.config)
        inputs, outputs = preloader.preload()

        image_width = self.config['input_config']['image_width']
        image_height = self.config['input_config']['image_height']
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

                volumes = [YoloVolume.from_raw_data(image_width, image_height, grid_size, num_classes, y.flatten().tolist())
                           for y in y_batch]
                y_batch = [vol.output_volume for vol in volumes]

                yield x_batch, np.array(y_batch)

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

        example = input.flatten().tolist() + output
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
