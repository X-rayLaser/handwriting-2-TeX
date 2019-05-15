from yolo.yolo_home import YoloDatasetHome
import os
import numpy as np


def train_on_precomputed_activations(regression_model, yolo_root, batch_size=8, epochs=100):
    yolo_home = YoloDatasetHome(yolo_root, input_dtype=np.float16)

    train_dataset_path = os.path.join(yolo_root, 'train')
    dev_dataset_path = os.path.join(yolo_root, 'dev')

    m_train = yolo_home.dataset_size(train_dataset_path)
    m_val = yolo_home.dataset_size(dev_dataset_path)
    print(m_train, m_val)

    generator = yolo_home.flow_with_preload(train_dataset_path,
                                            mini_batch_size=batch_size,
                                            normalize=False)

    dev_generator = yolo_home.flow_with_preload(dev_dataset_path,
                                                mini_batch_size=batch_size,
                                                normalize=True)

    regression_model.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=['mae'])

    regression_model.fit_generator(generator,
                                   steps_per_epoch=int(m_train / batch_size),
                                   epochs=epochs)


def end_to_end_model(feature_extractor, regression_model):
    from keras import Model
    from keras.layers import Input

    inp = Input(shape=feature_extractor.input_shape[1:])
    x = inp
    for layer in feature_extractor.layers:
        x = layer(x)

    for layer in regression_model.layers:
        x = layer(x)

    out = x

    combined_model = Model(inp, out)

    return combined_model


def validate_model(end_to_end_model, yolo_dataset):
    from yolo.draw_bounding_box import draw_boxes

    dataset_path = os.path.join(yolo_dataset, 'train')
    yolo = YoloDatasetHome(yolo_dataset)

    counter = 0

    # todo: do normalization here on batch
    for x_batch, y_batch in yolo.flow_with_preload(dataset_path, mini_batch_size=1, normalize=False):
        shape = (1,) + end_to_end_model.input_shape[1:]
        x = x_batch[0].reshape(shape) / 255.0
        pred_y = end_to_end_model.predict(x)
        draw_boxes(x_batch[0], 9, pred_y[0], p_treshold=0.1)
        counter += 1

        if counter > 3:
            break


def train():
    from models import get_feature_extractor, get_regression_model

    feature_extractor = get_feature_extractor(pretrained_model_path='../keras_model.h5')
    volume_shape = feature_extractor.output_shape[1:]

    regression_model = get_regression_model(input_shape=volume_shape, output_shape=(9, 9, 19))

    train_on_precomputed_activations(regression_model, '../datasets/yolo_precomputed')

    final_model = end_to_end_model(feature_extractor, regression_model)

    validate_model(final_model, '../datasets/yolo_dataset')


if __name__ == '__main__':
    train()
