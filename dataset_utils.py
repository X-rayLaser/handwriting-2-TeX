import os
import random
import csv
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from keras import utils


class_to_index = OrderedDict([
    ('0', 0),
    ('1', 1),
    ('2', 2),
    ('3', 3),
    ('4', 4),
    ('5', 5),
    ('6', 6),
    ('7', 7),
    ('8', 8),
    ('9', 9),
    ('+', 10),
    ('-', 11),
    ('times', 12),
    ('div', 13)
])

index_to_class = list(class_to_index.keys())


def examples_of_category(csv_files_dir, fname):
    class_name, _ = os.path.splitext(fname)
    category = class_to_index[class_name]

    path = os.path.join(csv_files_dir, fname)
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            yield map(np.uint8, row), category


def shuffle_data(x, y):
    m = y.shape[0]
    indices = list(range(m))
    random.shuffle(indices)
    x = x[indices]
    y = y[indices]
    return x, y


def dataset_generator(x, labels, mini_batch_size=128):
    while True:
        x, labels = shuffle_data(x, labels)

        for i in range(0, labels.shape[0], mini_batch_size):
            x_batch = x[i:i + mini_batch_size, :]
            y_batch = utils.to_categorical(labels[i:i + mini_batch_size], num_classes=14)

            x_batch_norm = x_batch.reshape((x_batch.shape[0], 45, 45, 1)) / 255.0
            yield x_batch_norm, y_batch


def all_examples_with_category(args):
    csv_dir, fname, category_counts = args
    class_name, _ = os.path.splitext(fname)
    category = class_to_index[class_name]
    m = category_counts[category]
    nx = 45 * 45

    x = np.zeros((m, nx), dtype=np.uint8)
    y = np.ones(m, dtype=np.uint8) * category

    i = 0
    for features, _ in examples_of_category(csv_dir, fname):
        x[i, :] = list(features)
        i += 1
        if i % 1000 == 0:
            print('Category "{}". Loaded {} / {}'.format(class_name, i, m))

    print('All {} examples of category "{}" are loaded in memory'.format(m, class_name))
    return x, y


def load_in_parallel(csv_files_dir, max_workers=4):
    m, category_counts = dataset_size(csv_files_dir)
    x = np.zeros((m, 45 * 45), dtype=np.uint8)
    y = np.zeros(m, dtype=np.uint8)

    i = 0

    pool = ProcessPoolExecutor(max_workers=max_workers)

    arg_list = [(csv_files_dir, fname, category_counts)
                for fname in os.listdir(csv_files_dir)]

    results = list(pool.map(all_examples_with_category, arg_list))

    for x_batch, y_batch in results:
        m = x_batch.shape[0]
        x[i:i + m, :] = x_batch
        y[i:i + m] = y_batch
        i += m

    print('All examples are loaded!')
    return x, y


def load_dataset(csv_files_dir):
    return load_in_parallel(csv_files_dir)


def dataset_size(csv_files_dir):
    m = 0

    num_classes = 14
    counts = [0] * num_classes
    for fname in os.listdir(csv_files_dir):
        class_name, _ = os.path.splitext(fname)
        category = class_to_index[class_name]
        path = os.path.join(csv_files_dir, fname)

        index = category
        with open(path, 'r', newline='') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                m += 1
                counts[index] += 1

    assert sum(counts) == m

    return m, counts
