from keras.layers import Input
from keras import Model


def one_layer_model(model, input_shape, layer_index):
    inp = Input(shape=input_shape)
    hidden_layer = model.layers[layer_index]
    output_layer = model.layers[-1]

    x = inp

    x = hidden_layer(x)
    out = output_layer(x)

    return Model(input=inp, output=out)


def pretrain_with_generator(data_generator, model, compilation_params, fitting_params):
    model.compile(**compilation_params)
    model.fit_generator(data_generator, **fitting_params)


def get_precomputed_generator(data_generator, shallow_model):
    for x_batch, y_batch in data_generator:
        features = shallow_model.predict(x_batch)
        yield features, y_batch


def build_complete_model(pretrained_layers, input_shape, output_layer):
    inp = Input(shape=input_shape)
    x = inp
    for layer in pretrained_layers:
        x = layer(x)

    out = output_layer(x)
    return Model(input=inp, output=out)


def fine_tune(data_generator, model, compilation_params, fitting_params):
    model.compile(**compilation_params)
    model.fit_generator(data_generator, **fitting_params)


def train_with_generator(data_generator, model, compilation_params, fitting_params):
    gen = data_generator
    pretrained_layers = []
    for i in range(len(model.layers) - 1):
        input_shape = model.input_shape
        shallow = one_layer_model(model, input_shape=input_shape,  layer_index=i)
        pretrain_with_generator(gen, shallow, compilation_params, fitting_params)

        hidden_layer = shallow.layers[0]
        pretrained_layers.append(hidden_layer)
        gen = get_precomputed_generator(gen, shallow)

    output_layer = shallow.layers[-1]
    complete_model =build_complete_model(pretrained_layers, output_layer)

    #fine_tune(data_generator, complete_model)
    return complete_model
