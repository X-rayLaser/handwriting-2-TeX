import numpy as np
from segmentation import extract_segments
from building_blocks import Primitive
import config
from dataset_utils import index_to_class


image_size = config.image_size


def image_to_latex(image, model):
    from construction import construct_latex
    segments = extract_segments(image)

    digits = recognize(segments, model)

    return construct_latex(digits, image.shape[1], image.shape[0])


def recognize(segments, model):
    res = []

    for segment in segments:
        x, y = segment.bounding_box.xy_center
        region = segment.bounding_box

        if segment.bounding_box.width > 45:
            res.append(Primitive('div', region))
        else:
            x_input = prepare_input(segment)
            A = model.predict(x_input)
            class_index = np.argmax(np.max(A, axis=0), axis=0)

            category_class = index_to_class[class_index]
            res.append(Primitive.new_primitive(category_class, x, y))

    return res


def prepare_input(segment):
    x = segment.pixels / 255.0
    return x.reshape(1, image_size, image_size, 1)


if __name__ == '__main__':
    import argparse
    from models import get_math_symbols_model
    from data_synthesis import Synthesizer

    parser = argparse.ArgumentParser(
        description='Test machine learning pipeline on'
                    'artificial math expression images'
    )
    parser.add_argument('--examples', type=float, default=2,
                        help='number of examples to test on')

    args = parser.parse_args()

    synth = Synthesizer('datasets/digits_and_operators_csv/test')

    model = get_math_symbols_model()

    n = args.examples

    correct = 0
    for i in range(n):
        image, latex = synth.synthesize_example()

        predicted_latex = image_to_latex(image, model)
        if latex == predicted_latex:
            correct += 1
        else:
            from data_synthesis import visualize_image
            visualize_image(image)
            print('Invalid recognition: {} -> {}'.format(latex,
                                                         predicted_latex))

    print('Classified correctly {} out of {} expressions'.format(correct, n))
