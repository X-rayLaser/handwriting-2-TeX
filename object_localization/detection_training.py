from object_localization.common import create_example_image, DistortedImagesGenerator


class ObjectDetectionGenerator(DistortedImagesGenerator):
    def __init__(self, csv_files_dir, mini_batch_size=32, output_width=45,
                 output_height=45, max_shift=2):
        super().__init__(csv_files_dir=csv_files_dir, num_classes=1,
                         mini_batch_size=mini_batch_size,
                         output_width=output_width,
                         output_height=output_height,
                         max_shift=max_shift,
                         p_background=0.5)

    def preprocess(self, x, y):
        x = self.distort_example(x, y)

        if self.decide_to_detect():
            shift = self.max_shift
            y = 1
        else:
            shift = self.output_height
            y = 0

        x = self.create_shifted_image(x, shift)

        return x, y


def detection_model(input_shape):
    from object_localization.overfeat import ModelBuilder
    builder = ModelBuilder(input_shape=input_shape, initial_num_filters=10)
    builder.add_conv_layer(last_one=True).add_batch_norm_layer()

    builder.add_fully_connected_layer().add_fully_connected_layer()
    builder.add_binary_classification_layer()

    return builder


if __name__ == '__main__':
    from dataset_utils import dataset_size
    from object_localization.common import train_model
    csv_dir = '../datasets/digits_and_operators_csv/train'
    csv_dev = '../datasets/digits_and_operators_csv/dev'

    m_train, _ = dataset_size(csv_dir)
    m_val, _ = dataset_size(csv_dev)

    mini_batch_size = 32

    distorted_generator = ObjectDetectionGenerator(csv_dir,
                                                   mini_batch_size=mini_batch_size)

    validation_generator = ObjectDetectionGenerator(csv_dev,
                                                    mini_batch_size=mini_batch_size)

    model_builder = detection_model(input_shape=(45, 45, 1))
    model = model_builder.get_complete_model(input_shape=(45, 45, 1))

    train_model(model, train_gen=distorted_generator.generate(),
                validation_gen=validation_generator.generate(),
                m_train=m_train, m_val=m_val, mini_batch_size=mini_batch_size,
                save_path='../detection_model.h5', epochs=2,
                loss='binary_crossentropy')
