import numpy as np
from keras.utils import to_categorical


def get_dictionary():
    d = dict((str(i), i) for i in range(10))
    d.update({
        ' ': 11,
        '+': 12,
        '-': 13,
        '*': 14,
        '\\': 15,
        '{': 16,
        '}': 17,
        '^': 18,
        'e': 19
    })
    return d


def char_to_vec(ch):
    class_index = get_dictionary()[ch]
    return to_categorical(class_index)


def vec_to_char(vec):
    class_index = np.argmax(vec)
    d = get_dictionary()

    for k, v in d.items():
        if v == class_index:
            return k


def tex_to_vecseq(tex):
    seq = []
    for ch in tex:
        seq.append(char_to_vec(ch))

    seq.append(char_to_vec('e'))
    return seq


def feature_vector_from_detections(detections, num_labels=14, image_width=250, image_height=250):

    labels_dict = dict((str(i), i) for i in range(10))
    labels_dict.update({
        '+': 10,
        '-': 11,
        'times': 12,
        'div': 13
    })

    seq = []
    for box, label in detections:
        x, y, w, h = box
        x = x / image_width
        y = y / image_height
        w = w / image_width
        h = h / image_height

        relative_box_vec = np.array([x, y, w, h])

        class_index = labels_dict[label]
        v = to_categorical(class_index, num_classes=num_labels)

        v = np.concatenate((relative_box_vec, v))
        seq.append(v)

    return seq


def gen(num_exaples=10):
    from data_synthesis import Synthesizer
    from object_localization.localization_pipeline import detect_objects
    from models import get_localization_model

    img_width = 250
    img_height = 250

    loc_model = get_localization_model(img_width=img_width,
                                       img_height=img_height,
                                       path='../localization_model.h5')

    csv_dir = '../datasets/digits_and_operators_csv/train'

    synthesizer = Synthesizer(csv_dir, img_width, img_height)

    x = []
    y = []

    for i in range(num_exaples):
        image, tex = synthesizer.synthesize_example()

        boxes, labels = detect_objects(image, loc_model)
        detection_results = list(zip(boxes, labels))
        input_sequence = feature_vector_from_detections(detection_results)
        output_sequence = tex_to_vecseq(tex)
        x.append(input_sequence)
        y.append(output_sequence)

    return x, y


def rnn_model():
    pass