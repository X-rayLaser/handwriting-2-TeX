import numpy as np
from segmentation import extract_segments
from building_blocks import Primitive
import config
from dataset_utils import index_to_class
from data_synthesis import visualize_image


image_size = config.image_size


def classifiy(image, model):
    return feed_x(image, model)


def detect_objects(image, detection_model, classification_model):
    from object_localization.localization_pipeline import detect_locations

    boxes, labels = detect_locations(image, detection_model,
                                     classification_model)

    assert len(boxes) == len(labels)

    res = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = labels[i]
        from building_blocks import RectangularRegion

        region = RectangularRegion(x, y, w, h)

        category_class = label
        res.append(Primitive(category_class, region))

    div_lines = get_division_lines(image)

    for div_line in div_lines:
        print('div line')
        res = [primitive for primitive in res if div_line.region.IoU(primitive.region) <= 0]

    return res + div_lines


def image_to_latex(image, detection_model, classification_model):
    from construction import construct_latex
    #segments = extract_segments(image)
    #primitives = recognize(segments, model)

    primitives = detect_objects(image, detection_model, classification_model)

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
    from object_localization.detection_training import detection_model

    img_width = 400
    img_height = 300

    parser = argparse.ArgumentParser(
        description='Test machine learning pipeline on'
                    'artificial math expression images'
    )
    parser.add_argument('--examples', type=float, default=25,
                        help='number of examples to test on')

    args = parser.parse_args()

    synth = Synthesizer('datasets/digits_and_operators_csv/test', img_width=img_width, img_height=img_height)

    #model = get_math_symbols_model()
    from object_localization.localization_training import model as classification_model

    model = classification_model(input_shape=(45, 45, 1), num_classes=14)
    model.load_weights('localization_model.h5')

    dmodel_builder = detection_model(input_shape=(45, 45, 1))
    dmodel_builder.load_weights('detection_model.h5')
    det_model = dmodel_builder.get_complete_model(input_shape=(img_height, img_width, 1))

    n = args.examples

    correct = 0
    predicted_latex = ''
    latex = ''
    for i in range(n):
        try:
            image, latex = synth.synthesize_example()

            predicted_latex = image_to_latex(image, det_model, model)
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
            #visualize_image(image)
            print('Invalid recognition: {} -> {}'.format(latex,
                                                         predicted_latex))

    percentage = float(correct) / n * 100
    print('Classified correctly {} %, ({} out of {} expressions)'.format(
        percentage, correct, n)
    )
