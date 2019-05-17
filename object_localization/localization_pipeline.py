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


def non_max_suppression(boxes, probs):
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
            return IoU(box, b) < 0.1

        rems = list(filter(small_iou, rems))

    return survived_boxes, survived_scores


def detect_category(y_pred, class_index, width, height, num_classes=15):
    y_pred = y_pred[:, :, class_index]
    p_threshold = 0.6
    mask = y_pred > p_threshold

    y_pred = y_pred * mask

    boxes = []
    scores = []

    print(y_pred.shape)
    rows, cols = y_pred.shape

    for row in range(rows):
        for col in range(cols):
            if y_pred[row, col] > p_threshold:
                x = int(round(col / cols * width))
                y = int(round(row / rows * height))

                xc = x + 45 // 2
                yc = y + 45 // 2

                boxes.append((x, y, 45, 45))
                scores.append(y_pred[row, col])

    return non_max_suppression(boxes, scores)


def detect_objects():
    from data_synthesis import ImagesGenerator, visualize_image, Synthesizer
    from object_localization.localization_training import model
    from dataset_utils import dataset_size
    import numpy as np
    csv_dir = '../datasets/digits_and_operators_csv/train'

    m_train, _ = dataset_size(csv_dir)

    img_width = 600
    img_height = 500
    localization_model = model(input_shape=(img_height, img_width, 1), num_classes=15)
    localization_model.load_weights('../localization_model.h5')

    csv_dir_test = '../datasets/digits_and_operators_csv/test'

    synthesizer = Synthesizer(csv_dir_test, img_width=img_width, img_height=img_height)

    img, latex = synthesizer.synthesize_example()

    y_pred = localization_model.predict(img.reshape(1, img_height, img_width, 1) / 255.0)

    output_shape = localization_model.output_shape[1:]
    y_pred = y_pred.reshape(output_shape)
    print(y_pred.shape)
    print(np.argmax(y_pred))

    all_boxes = []
    all_labels = []
    all_scores = []
    from dataset_utils import index_to_class
    for k in range(14):
        boxes, scores = detect_category(y_pred, k, width=img_width, height=img_height)
        all_boxes.extend(boxes)
        all_scores.extend(scores)
        all_labels.extend([index_to_class[k]] * len(boxes))

    all_boxes2, all_scores2 = non_max_suppression(all_boxes, all_scores)

    all_labels2 = []
    for b in all_boxes2:
        index = all_boxes.index(b)
        if index == -1:
            raise Exception('3')
        all_labels2.append(all_labels[index])

    from yolo.draw_bounding_box import visualize_detection
    visualize_detection(img, all_boxes, all_labels)

    visualize_detection(img, all_boxes2, all_labels2)


if __name__ == '__main__':
    detect_objects()
