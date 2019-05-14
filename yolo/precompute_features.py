from yolo.yolo_home import YoloDatasetHome


def precompute_features(feature_extractor, dataset_root, destination):
    yolo_source = YoloDatasetHome(dataset_root)

    m = yolo_source.config['num_examples']
    input_config = yolo_source.config['input_config']
    output_config = yolo_source.config['output_config']
    training_fraction = yolo_source.config['training_fraction']
    number_of_parts = yolo_source.config['number_of_parts']

    precomputed_input_config = dict(input_config)
    precomputed_input_config['shape'] = feature_extractor.output_shape[1:]
    yolo_destination = YoloDatasetHome.initialize_dataset(destination,
                                                          num_examples=m,
                                                          input_config=precomputed_input_config,
                                                          output_config=output_config,
                                                          training_fraction=training_fraction,
                                                          number_of_parts=number_of_parts)

    import numpy as np
    w = input_config['image_width']
    h = input_config['image_height']
    for x, y in yolo_source.get_all_examples():
        inp = np.array(x).reshape((1, w, h, 1)) / 255.0
        feature_maps = feature_extractor.predict(inp)[0]
        yolo_destination.add_example(feature_maps, y)


if __name__ == '__main__':
    from models import get_feature_extractor

    yolo_source = '../datasets/yolo_dataset'
    destination_dir = '../datasets/yolo_precomputed'

    extractor = get_feature_extractor(pretrained_model_path='../keras_model.h5')
    precompute_features(extractor, yolo_source, destination_dir)
