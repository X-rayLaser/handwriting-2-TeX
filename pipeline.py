import numpy as np
from segmentation import extract_segments
from building_blocks import Primitive
import config
from dataset_utils import index_to_class
from data_synthesis import visualize_image


image_size = config.image_size


def classifiy(image, model):
    return feed_x(image, model)


def detect_objects(image, model):
    from object_localization import localization_pipeline as locpipe

    boxes, labels = locpipe.detect_objects(image, model)

    assert len(boxes) == len(labels)

    res = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = labels[i]
        from building_blocks import RectangularRegion

        region = RectangularRegion(x, y, w, h)

        if region.width > 70:
            res.append(Primitive('div', region))
        elif region.width > 45 and region.height < image_size / 8:
            res.append(Primitive('div', region))
        else:
            category_class = label
            res.append(Primitive(category_class, region))

    div_lines = get_division_lines(image)

    for div_line in div_lines:
        res = [primitive for primitive in res if primitive not in div_line.region]

    return res


def image_to_latex(image, model):
    from construction import construct_latex
    #segments = extract_segments(image)
    #primitives = recognize(segments, model)

    primitives = detect_objects(image, model)

    return construct_latex(primitives, image.shape[1], image.shape[0])


def feed_x(x, model):
    x_input = prepare_input(x)
    A = model.predict(x_input)
    class_index = np.argmax(np.max(A, axis=0), axis=0)

    category_class = index_to_class[class_index]

    return category_class


def recognize(segments, model):
    res = []

    for segment in segments:
        region = segment.bounding_box

        if segment.bounding_box.width > 70:
            res.append(Primitive('div', region))
        elif segment.bounding_box.width > 45 and segment.bounding_box.height < image_size / 8:
            res.append(Primitive('div', region))
        else:
            category_class = feed_x(segment.pixels, model)
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


if __name__ == '__main__':
    import argparse
    from models import get_math_symbols_model
    from data_synthesis import Synthesizer

    parser = argparse.ArgumentParser(
        description='Test machine learning pipeline on'
                    'artificial math expression images'
    )
    parser.add_argument('--examples', type=float, default=25,
                        help='number of examples to test on')

    args = parser.parse_args()

    synth = Synthesizer('datasets/digits_and_operators_csv/test')

    model = get_math_symbols_model()

    n = args.examples

    correct = 0
    predicted_latex = ''
    latex = ''
    for i in range(n):
        try:
            image, latex = synth.synthesize_example()

            predicted_latex = image_to_latex(image, model)
            if latex == predicted_latex:
                correct += 1
                #visualize_image(image)
            else:
                visualize_image(image)
                print('Invalid recognition: {} -> {}'.format(latex,
                                                             predicted_latex))
        except Exception:
            import traceback
            traceback.print_exc()
            visualize_image(image)
            print('Invalid recognition: {} -> {}'.format(latex,
                                                         predicted_latex))

    percentage = float(correct) / n * 100
    print('Classified correctly {} %, ({} out of {} expressions)'.format(
        percentage, correct, n)
    )
