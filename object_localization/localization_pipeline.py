def IoU(box1, box2):
    from shapely.geometry import box

    xc1, yc1, w1, h1 = box1
    xc2, yc2, w2, h2 = box2

    x1 = xc1 - 45 // 2
    y1 = yc1 - 45 // 2

    x2 = xc2 - 45 // 2
    y2 = yc2 - 45 // 2
    b1 = box(x1, y1, x1 + w1, y1 + h1)
    b2 = box(x2, y2, x2 + w2, y2 + h2)

    intersection = b1.intersection(b2).area

    union = b1.union(b2).area

    if union == 0:
        return 1

    return intersection / union


def non_max_suppression(boxes, probs, p_threshold=0.1):
    pairs = list(zip(boxes, probs))
    pairs.sort(key=lambda t: t[1])

    rems = list(pairs)
    survived_boxes = []
    survived_scores = []
    while rems:
        box, prob = rems.pop()
        survived_boxes.append(box)
        survived_scores.append(prob)

        def small_iou(t):
            b, p = t
            return IoU(box, b) < p_threshold

        rems = list(filter(small_iou, rems))

    return survived_boxes, survived_scores


def detect_boxes(prediction_grid, width, height, p_threshold=0.9):
    mask = prediction_grid > p_threshold

    prediction_grid = prediction_grid * mask

    boxes = []
    scores = []

    rows, cols = prediction_grid.shape

    for row in range(rows):
        for col in range(cols):
            if prediction_grid[row, col] > p_threshold:
                x = col + 45 // 2
                y = row + 45 // 2

                boxes.append((x, y, 45, 45))
                scores.append(prediction_grid[row, col])

    return boxes, scores


def cropped_areas(image, box):
    height, width = image.shape
    pixel_shift = 1
    xc, yc, w, h = box

    for i in range(-2, 2, 1):
        for j in range(-2, 2, 1):
            delta_x = i * pixel_shift
            delta_y = j * pixel_shift
            x = xc - 45 // 2
            y = yc - 45 // 2

            col = int(round(x + delta_x))
            row = int(round(y + delta_y))

            col = min(width - w - 1, max(0, col))
            row = min(height - h - 1, max(0, row))

            a = image[row:row+h, col:col+w]
            yield a


def recognize_object(image, bounding_box, classifier):
    from dataset_utils import index_to_class
    import numpy as np

    outputs = []
    counts = [0] * 14
    for image_patch in cropped_areas(image, bounding_box):
        a = image_patch.reshape((1, 45, 45, 1)) / 255.0
        y_hat = classifier.predict(a)[0]

        y_hat = np.squeeze(y_hat)
        cat_class = y_hat[:14].argmax()

        counts[cat_class] += 1
        outputs.append(y_hat[:14])

    import numpy as np
    v = np.mean(outputs, axis=0)

    return index_to_class[np.argmax(counts)]


def detect_locations(image, dmodel, classifier_model):
    img_height, img_width = image.shape

    y_pred = dmodel.predict(image.reshape(1, img_height, img_width, 1) / 255.0)

    output_shape = dmodel.output_shape[1:]
    y_pred = y_pred.reshape(output_shape)[:, :, 0]
    boxes, scores = detect_boxes(y_pred, img_width, img_height)
    boxes, scores = non_max_suppression(boxes, scores)

    labels = []
    for box in boxes:
        predicted_class = recognize_object(image, box, classifier_model)
        labels.append(predicted_class)
    return boxes, labels


if __name__ == '__main__':
    from data_synthesis import Synthesizer
    from object_localization.localization_training import model
    from object_localization.detection_training import detection_model

    img_width = 200
    img_height = 200

    csv_dir_test = '../datasets/digits_and_operators_csv/test'

    synthesizer = Synthesizer(csv_dir_test, img_width=img_width, img_height=img_height)

    localization_model = model(input_shape=(45, 45, 1), num_classes=14)
    localization_model.load_weights('../localization_model.h5')

    image, latex = synthesizer.synthesize_example()

    dmodel_builder = detection_model(input_shape=(45, 45, 1))
    dmodel_builder.load_weights('../detection_model.h5')
    dmodel = dmodel_builder.get_complete_model(input_shape=(img_height, img_width, 1))

    bounding_boxes, labels = detect_locations(image, dmodel, localization_model)

    from yolo.draw_bounding_box import visualize_detection
    visualize_detection(image, bounding_boxes, labels)
