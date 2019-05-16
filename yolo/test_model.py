import os
from models import end_to_end_model
from yolo.draw_bounding_box import draw_boxes


def validate_model(end_to_end_model, yolo_dataset, val_folder='test'):
    yolo_home = YoloDatasetHome(dataset_root=yolo_dataset)
    grid_size = yolo_home.config['output_config']['grid_size']

    dataset_path = os.path.join(yolo_dataset, val_folder)
    yolo = YoloDatasetHome(yolo_dataset)

    counter = 0

    for x_batch, y_batch in yolo.flow_with_preload(dataset_path, mini_batch_size=1, normalize=False):
        shape = (1,) + end_to_end_model.input_shape[1:]
        x = x_batch[0].reshape(shape) / 255.0
        pred_y = end_to_end_model.predict(x)
        draw_boxes(x_batch[0], grid_size, pred_y[0], p_treshold=0.1)
        counter += 1

        if counter > 10:
            break


if __name__ == '__main__':
    from yolo.yolo_home import YoloDatasetHome
    from models import get_feature_extractor, get_regression_model

    yolo_root_path = '../datasets/yolo_dataset'
    weights_path = '../yolo_model.h5'

    yolo_home = YoloDatasetHome(dataset_root=yolo_root_path)
    grid_size = yolo_home.config['output_config']['grid_size']
    num_classes = yolo_home.config['output_config']['num_classes']
    w = yolo_home.config['input_config']['image_width']
    h = yolo_home.config['input_config']['image_height']

    feature_extractor = get_feature_extractor(
        image_width=w, image_height=h, pretrained_model_path='../keras_model.h5'
    )
    volume_shape = feature_extractor.output_shape[1:]

    regression_model = get_regression_model(
        input_shape=volume_shape,
        output_shape=(grid_size, grid_size, 5 + num_classes)
    )
    final_model = end_to_end_model(feature_extractor, regression_model)

    final_model.load_weights(weights_path)

    validate_model(final_model, '../datasets/yolo_dataset', val_folder='train')

