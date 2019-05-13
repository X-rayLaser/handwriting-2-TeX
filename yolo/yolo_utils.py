def total_size(shape):
    n = 1

    for dim_size in shape:
        n *= dim_size

    return n


def get_input_size(config):
    input_shape = config['input_config']['shape']
    return total_size(input_shape)


def get_output_size(config):
    output_shape = config['output_config']['shape']
    return total_size(output_shape)

