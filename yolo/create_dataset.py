import os
import numpy as np
from data_synthesis import visualize_image
from yolo.yolo_home import YoloDatasetHome
from yolo.data_generator import YoloDataGenerator


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
    image_width = 150
    image_height = 150
    grid_size = 5
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
        n = 6
        input, volume = gen.make_example(elements=n)
        output = volume.to_raw_data()
        yolo_home.add_example(input, output)
        print('Created examples {} out of {}'.format(i, num_examples))


if __name__ == '__main__':
    destination_dir = '../datasets/yolo_dataset'
    csv_dir = '../datasets/digits_and_operators_csv/train'
    generate_dataset(primitives_source=csv_dir, destination_dir=destination_dir, num_examples=1000)
