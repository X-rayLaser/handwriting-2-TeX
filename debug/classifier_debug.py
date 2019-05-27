import numpy as np


def group_errors(ground_true, predictions):
    mistakes = {}

    total_mistakes = 0
    total_examples = len(predictions)

    for i in range(total_examples):
        y = ground_true[i]
        y_hat = predictions[i]
        if y != y_hat:
            if y not in mistakes:
                mistakes[y] = 0

            mistakes[y] += 1
            total_mistakes += 1

    error = total_mistakes / float(total_examples)

    sorted_mistakes = sorted(mistakes.items(), key=lambda t: t[1], reverse=True)

    return sorted_mistakes, error


def evaluate_classifier(classifier, height=45, width=45):
    from dataset_utils import index_to_class
    from generators.uniform import uniform_flow
    csv_dir = '../datasets/digits_and_operators_csv/dev'

    total_mistakes = 0
    ground_truths = []
    predictions = []

    for x_batch, y_batch in uniform_flow(csv_dir, num_classes=14,
                                         image_width=width,
                                         image_height=height, batch_size=1):
        x = x_batch
        y_hat = classifier.predict(x).squeeze()

        ground_true = index_to_class[y_batch.squeeze().argmax()]
        prediction = index_to_class[y_hat.argmax()]

        ground_truths.append(ground_true)
        predictions.append(prediction)

        if ground_true != prediction:
            total_mistakes += 1
            print('Error: {} -> {}'.format(ground_true, prediction))

        if total_mistakes > 10:
            break

    return group_errors(ground_truths, predictions)


if __name__ == '__main__':
    from object_localization.detection_training import detection_model
    from object_localization.localization_training import build_classification_model

    builder = build_classification_model(input_shape=(45, 45, 1), num_classes=14)
    builder.load_weights('../localization_model.h5')
    loc_model = builder.get_complete_model(input_shape=(45, 45, 1))
    #from models import initialize_math_recognition_model
    #loc_model = initialize_math_recognition_model()
    #loc_model.load_weights('../keras_model.h5')

    print('Testing classifier model on well cropped images of size 45x45:')
    mistakes, error_rate = evaluate_classifier(loc_model)

    print('Error rate is {} %'.format(error_rate * 100))
    print('Categories got wrong most often:')
    print(mistakes)
