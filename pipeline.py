import numpy as np
from segmentation import extract_segments
from building_blocks import Primitive
import config
from dataset_utils import index_to_class
from data_synthesis import visualize_image


image_size = config.image_size


def image_to_latex(image, classification_model):
    from construction import construct_latex
    segments = extract_segments(image)
    primitives = recognize(segments, classification_model)

    return construct_latex(primitives, image.shape[1], image.shape[0])


def recognize(segments, model):
    res = []

    for segment in segments:
        region = segment.bounding_box

        if segment.bounding_box.width > 70:
            res.append(Primitive('div', region))
        elif segment.bounding_box.width > 45 and segment.bounding_box.height < image_size / 8:
            res.append(Primitive('div', region))
        else:
            x = segment.pixels
            x_input = prepare_input(x)
            a = model.predict(x_input)
            category_class = index_to_class[np.argmax(a)]
            res.append(Primitive(category_class, segment.bounding_box))

    return res


def get_division_lines(image):
    segments = extract_segments(image)
    res = []

    for segment in segments:
        region = segment.bounding_box

        if segment.bounding_box.width > 70:
            res.append(Primitive('div', region))
        elif segment.bounding_box.width > 45 and segment.bounding_box.height < image_size / 8:
            res.append(Primitive('div', region))

    return res


def prepare_input(x):
    x = x / 255.0
    return x.reshape(1, image_size, image_size, 1)
