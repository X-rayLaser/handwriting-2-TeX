import csv
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .yolo_utils import get_output_size, get_input_size


def load_part(args):
    path, num_examples, input_size, output_size = args

    dtype = np.uint8
    x_batch = np.zeros((num_examples, input_size), dtype=dtype)
    y_batch = np.zeros((num_examples, output_size), dtype=np.uint16)

    i = 0
    with open(path, 'r', newline='') as f:
        for row in csv.reader(f):
            xy = list(map(np.uint16, row))
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
        y = np.zeros((m, self._output_size), dtype=np.uint16)

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
        # todo: fix this hardcoding
        return 200 * 5 + 1
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
