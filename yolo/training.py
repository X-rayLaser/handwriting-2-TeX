from yolo.yolo_home import YoloDatasetHome
import os


def load_pretrained_model(path):
    from models import get_math_symbols_model
    model = get_math_symbols_model()

    return model


def chomp_off_last_layers(model, n=4):
    for i in range(n):
        model.layers.pop()

    model.summary(line_length=150)
    return model


def precompute_activations(model, dataset):
    pass


def create_regression_model(input_shape, output_shape):
    pass


def train_on_precomputed_activations(regression_model, yolo_root, batch_size=64, epochs=5):
    yolo_home = YoloDatasetHome(yolo_root)

    train_dataset_path = os.path.join(yolo_root, 'train')
    dev_dataset_path = os.path.join(yolo_root, 'dev')

    m_train = yolo_home.dataset_size(train_dataset_path)
    m_val = yolo_home.dataset_size(dev_dataset_path)

    generator = yolo_home.flow_with_preload(train_dataset_path,
                                            mini_batch_size=batch_size,
                                            normalize=True)

    dev_generator = yolo_home.flow_with_preload(dev_dataset_path,
                                                mini_batch_size=batch_size,
                                                normalize=True)

    regression_model.compile(optimizer='adam', loss='mean_squared_error',
                             metrics=['mae'])

    regression_model.fit_generator(generator,
                                   steps_per_epoch=int(m_train / batch_size),
                                   epochs=epochs,
                                   validation_data=dev_generator,
                                   validation_steps=int(m_val / batch_size))


def end_to_end_model(feature_extractor, regression_model):
    from keras import Model
    feature_extractor.summary(line_length=150)

    inp = feature_extractor.input
    x = inp
    for layer in feature_extractor.layers:
        x = layer(x)

    for layer in regression_model.layers:
        x = layer(x)

    out = x

    combined_model = Model(inp, out)
    combined_model.summary(line_length=150)

    return combined_model


def validate_model(end_to_end_model, yolo_dataset):
    pass


def train():
    from models import get_feature_extractor, get_regression_model
    from yolo.precompute_features import precompute_features

    feature_extractor = get_feature_extractor()
    volume_shape = feature_extractor.output_shape[1:]

    regression_model = get_regression_model(input_shape=volume_shape, output_shape=(9, 9, 19))

    train_on_precomputed_activations(regression_model, '../datasets/yolo_precomputed')

    final_model = end_to_end_model(feature_extractor, regression_model)

    
    final_model
    #validate_model(final_model, '')
