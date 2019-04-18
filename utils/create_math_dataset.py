import os
import random
import csv
import numpy as np
from PIL import Image
from PIL import ImageOps


im_size = 45
m_train = 149618.0


def random_destination_directory(destination, train_data_fraction, count_dict):
    train_destination = os.path.join(destination, 'train')
    dev_destination = os.path.join(destination, 'dev')
    test_destination = os.path.join(destination, 'test')

    os.makedirs(train_destination, exist_ok=True)
    os.makedirs(dev_destination, exist_ok=True)
    os.makedirs(test_destination, exist_ok=True)

    prob_train = train_data_fraction
    prob_dev = (1 - prob_train) / 2.0
    prob_test = 1 - prob_train

    r = random.random()
    if r < prob_dev:
        count_dict['dev'] += 1
        return dev_destination
    elif r < prob_test:
        count_dict['test'] += 1
        return test_destination
    else:
        count_dict['train'] += 1
        return train_destination


def create_csv_dataset(dataset_root_path, destination, train_data_fraction=0.9):
    n = 0
    count_dict = {
        'train': 0,
        'dev': 0,
        'test': 0
    }

    for cat_dir in os.listdir(dataset_root_path):
        category = cat_dir
        cat_path = os.path.join(dataset_root_path, cat_dir)

        for fname in os.listdir(cat_path):
            fpath = os.path.join(cat_path, fname)
            im = Image.open(fpath).convert('L')
            im = ImageOps.invert(im)
            pixels = np.array(im.getdata(), dtype=np.uint8).reshape(im_size * im_size)

            destination_folder = random_destination_directory(destination,
                                                              train_data_fraction,
                                                              count_dict)
            dest_path = os.path.join(destination_folder, '{}.csv'.format(category))
            with open(dest_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pixels.tolist())

            n += 1

        print('Created category', category, 'Created examples:', n)

    print('Created examples total:', n)
    print('Training set size:', count_dict['train'])
    print('Development set size:', count_dict['dev'])
    print('Test set size:', count_dict['test'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Given an image, compress it using JPEG algorithm'
    )
    parser.add_argument('source_dir', type=str,
                        help='absolute path to the dataset')

    parser.add_argument('destination_dir', type=str,
                        help='absolute path to the destination')

    args = parser.parse_args()

    create_csv_dataset(args.source_dir, args.destination_dir)
