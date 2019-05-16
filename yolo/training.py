from yolo.yolo_home import YoloDatasetHome
import os
import numpy as np
from models import get_feature_extractor, get_regression_model, end_to_end_model


def train_on_precomputed_activations(regression_model, yolo_root, batch_size=32, epochs=20):
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
                                                normalize=False)

    from keras.optimizers import Adam

    adam = Adam(lr=0.001)

    def custom_loss(y_true, y_pred):
        print(y_true.shape, y_pred.shape)
        return 0

    regression_model.compile(optimizer=adam, loss='mean_squared_error',
                             metrics=['mae'])

    regression_model.fit_generator(generator,
                                   steps_per_epoch=int(m_train / batch_size),
                                   epochs=epochs)


def train():
    yolo_home = YoloDatasetHome(dataset_root='../datasets/yolo_dataset')
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

    train_on_precomputed_activations(regression_model, '../datasets/yolo_precomputed',
                                     epochs=20)

    final_model = end_to_end_model(feature_extractor, regression_model)

    final_model.save_weights('../yolo_model.h5')


if __name__ == '__main__':
    train()
