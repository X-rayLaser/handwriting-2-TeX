import numpy as np
from object_localization.localization_pipeline import IoU
from yolo.data_generator import YoloDataGenerator
from models import get_localization_model
from object_localization.localization_pipeline import detect_objects
from yolo.draw_bounding_box import visualize_detection


def localization_score(box, predicted_box):
    pass


def classification_score(label, predicted_label):
    pass


def metric_score(correct_answers, predictions):
    def sort_func(t):
        box, label = t
        x, y, w, h = box
        return x, y

    correct_answers = list(correct_answers)
    predictions = list(predictions)

    correct_answers.sort(key=sort_func)
    predictions.sort(key=sort_func)

    ious = []
    misclassified = 0

    from dataset_utils import class_to_index
    for correct_box, correct_label in correct_answers:
        max_iou = -0.00000001
        nearest_label = None
        best_prediction = None
        for prediction in predictions:
            predicted_box, predicted_label = prediction
            iou = IoU(correct_box, predicted_box)
            if iou > max_iou:
                max_iou = iou
                nearest_label = class_to_index[predicted_label]
                best_prediction = prediction

        ious.append(max_iou)
        if nearest_label is None or nearest_label != correct_label:
            misclassified += 1

        if best_prediction is not None:
            predictions.remove(best_prediction)

    iou_score = np.mean(ious)
    accuracy_score = 1.0 - misclassified / len(correct_answers)
    return iou_score * accuracy_score


def evaluate(image_width=300, image_height=300, num_examples=100, objects_per_image=10):
    primitives_source = '../datasets/digits_and_operators_csv/train'

    gen = YoloDataGenerator(img_width=image_width, img_height=image_height,
                            primitives_dir=primitives_source,
                            grid_size=9, num_classes=14)

    loc_model = get_localization_model(img_width=image_width,
                                       img_height=image_height,
                                       path='../localization_model.h5')

    scores = []
    left_overs = 0
    for i in range(num_examples):
        inp, volume = gen.make_example(elements=objects_per_image)
        image = inp
        bounding_boxes = volume.boxes
        labels = volume.classes

        correct_answers = list(zip(bounding_boxes, labels))

        box_predictions, class_predictions = detect_objects(image, loc_model)
        predictions = list(zip(box_predictions, class_predictions))
        score = metric_score(correct_answers, predictions)
        scores.append(score)
        left_overs += abs(len(predictions) - len(correct_answers))

        visualize_detection(image, box_predictions, class_predictions)

    mean_score = np.mean(scores)
    print('Optmizing metric: {}'.format(mean_score))
    print('Average fraction of left-overs: {}'.format(left_overs / num_examples))


if __name__ == '__main__':
    evaluate(num_examples=5)
