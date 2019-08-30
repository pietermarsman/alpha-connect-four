import json
import os
from random import sample
from tempfile import NamedTemporaryFile

import numpy as np
from tensorflow.python.keras import Input, Model, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.python.keras.layers import Dense, Conv3D, Flatten, Reshape, \
    RepeatVector, Permute, BatchNormalization, Concatenate, ReLU, Softmax
from tensorflow.python.keras.optimizers import Adam
from tqdm import tqdm

from observer import AlphaConnectSerializer
from state import State, FOUR, Action, Augmentation
from util import list_files, winner_value


def train_new_model(data_path, log_path=None, max_games=None):
    K.clear_session()
    input_shape = State.empty().to_numpy().shape[-1]
    model = create_model(input_shape, filters=12)
    print(model.summary())

    if data_path is not None:
        x_state, y_policy, y_reward = read_data(data_path, max_games)
        callbacks = [EarlyStopping(patience=5)]
        if log_path is not None:
            callbacks.append(CSVLogger(log_path))
        model.fit(x_state, [y_policy, y_reward], epochs=100, validation_split=0.3, callbacks=callbacks)

    return model


def read_data(data_path, max_games=None):
    game_files = list(sorted(list_files(data_path, '.json')))
    augmentations = list(Augmentation.iter_augmentations())

    if max_games is not None:
        game_files = list(game_files)[-max_games:]
        print('Using game files from %s to %s' % (game_files[0], game_files[-1]))

    x = []
    y_policy = []
    y_reward = []

    for game_path in tqdm(game_files):
        with open(game_path, 'r') as fin:
            game_data = json.load(fin)

        winner, starter, actions, policies = AlphaConnectSerializer.deserialize(game_data)

        state = State.empty()
        states = [state]
        for action in actions:
            state = state.take_action(action)
            states.append(state)
        states, final_state = states[:-1], states[-1]

        n_samples = min(len(states), len(augmentations))
        game_samples = sample(list(range(len(states))), n_samples)
        for augmentation, i in zip(augmentations, game_samples):
            augmentend_action_order = sorted(Action.iter_actions(), key=lambda a: a.augment(augmentation).to_int())

            x.append(states[i].to_numpy(augmentation))
            y_policy.append([policies[i].get(action, 0.0) for action in augmentend_action_order])
            y_reward.append(winner_value(final_state.winner, states[i]))

    return np.array(x), np.array(y_policy), np.array(y_reward)


def create_model(input_size, filters, c=10 ** -4):
    l2 = regularizers.l2(c)
    input = Input(shape=(FOUR, FOUR, FOUR, input_size))

    conv_1 = normalized_relu(line_convolution(input, filters, l2))
    conv_2 = normalized_relu(line_convolution(conv_1, filters, l2))
    conv_3 = normalized_relu(line_convolution(conv_2, filters, l2))
    last_conv = Conv3D(filters, 1, kernel_regularizer=l2)(conv_3)

    collapse_play = normalized_relu(Conv3D(filters // 2, (1, 1, 4), kernel_regularizer=l2)(last_conv))
    squash_play = normalized_relu(Conv3D(3, 1, kernel_regularizer=l2)(collapse_play))
    flatten_play = Flatten()(squash_play)
    dense_play = Dense(16, activation='relu', kernel_regularizer=l2)(flatten_play)
    sigmoid_play = Dense(16, activation='sigmoid', kernel_regularizer=l2)(dense_play)
    output_play = Softmax()(sigmoid_play)

    collapse_win = normalized_relu(Conv3D(3, (1, 1, 4), kernel_regularizer=l2)(last_conv))
    flatten_win = Flatten()(collapse_win)
    dense_win = Dense(16, activation='relu', kernel_regularizer=l2)(flatten_win)
    output_win = Dense(1, activation='tanh', kernel_regularizer=l2)(dense_win)

    model = Model(inputs=input, outputs=[output_play, output_win])
    optimizer = Adam()
    metics = {'softmax': 'categorical_accuracy', 'dense_3': 'mae'}
    model.compile(optimizer, ['categorical_crossentropy', 'mse'], metrics=metics)
    return model


def line_convolution(input, filters, l2):
    conv = Conv3D(filters, 1, kernel_regularizer=l2)(input)
    permute_x1, permute_y1 = horizontal_axis_convolution(conv, filters // 3, l2)
    permute_z1 = vertical_axix_convolution(conv, filters // 3, l2)
    permute_xy = horizontal_plane_convolution(conv, filters // 4, l2)
    permute_xz, permute_yz = vertical_plane_convolution(conv, filters // 4, l2)
    permute_xyz = box_convolution(conv, filters // 6, l2)
    concatenate = Concatenate()([conv, permute_x1, permute_y1, permute_z1, permute_xy, permute_xz, permute_yz,
                                 permute_xyz])
    return concatenate


def horizontal_axis_convolution(input, filters, l2):
    shared_squash = Conv3D(filters, [4, 1, 1], kernel_regularizer=l2)

    squash_x = shared_squash(input)
    box_x = spread_plane(squash_x, filters, [1, 2, 3, 4])  # x, y, z, f -> x, y, z, f

    rotate = Permute([2, 1, 3, 4])(input)
    squash_y = shared_squash(rotate)
    box_y = spread_plane(squash_y, filters, [2, 1, 3, 4])  # y, x, z, f -> x, y, z, f

    return box_x, box_y


def vertical_axix_convolution(input, filters, l2):
    squash_z = Conv3D(filters, [1, 1, 4], kernel_regularizer=l2)(input)
    box_z = spread_plane(squash_z, filters, [2, 3, 1, 4])  # z, x, y, f -> x, y, z, f
    return box_z


def spread_plane(input, filters, permute_dims):
    gather = Reshape((FOUR * FOUR * filters,))(input)
    repeat = RepeatVector(FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, filters))(repeat)
    permute = Permute(permute_dims)(spread)
    return permute


def vertical_plane_convolution(input, plane_filters, l2):
    reduce = Conv3D(plane_filters, 1, kernel_regularizer=l2)(input)
    shared_squash = Conv3D(plane_filters, [4, 1, 4], kernel_regularizer=l2)

    squash_xz = shared_squash(reduce)
    box_xz = spread_axis(squash_xz, plane_filters, [1, 3, 2, 4])  # x, z, y, f -> x, y, z, f

    rotate = Permute([2, 1, 3, 4])(reduce)
    squash_yz = shared_squash(rotate)
    box_yz = spread_axis(squash_yz, plane_filters, [3, 1, 2, 4])  # y, z, x, f -> x, y, z, f

    return box_xz, box_yz


def horizontal_plane_convolution(input, plane_filters, l2):
    reduce = Conv3D(plane_filters, 1, kernel_regularizer=l2)(input)
    squash = Conv3D(plane_filters, [4, 4, 1], kernel_regularizer=l2)(reduce)
    permute = spread_axis(squash, plane_filters, [1, 2, 3, 4])  # x, y, z, f -> x, y, z, f
    return permute


def spread_axis(input, filters, permute_dims):
    gather = Reshape((FOUR * filters,))(input)
    repeat = RepeatVector(FOUR * FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, filters))(repeat)
    permute = Permute(permute_dims)(spread)
    return permute


def box_convolution(input, box_filters, l2):
    reduce = Conv3D(box_filters, 1, kernel_regularizer=l2)(input)
    squash = Conv3D(box_filters, (FOUR, FOUR, FOUR), kernel_regularizer=l2)(reduce)
    gather = Reshape((box_filters,))(squash)
    repeat = RepeatVector(FOUR * FOUR * FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, box_filters))(repeat)
    return spread


def normalized_relu(layer):
    norm = BatchNormalization()(layer)
    relu = ReLU()(norm)
    return relu


def write_model(model, model_path):
    with NamedTemporaryFile(dir=os.path.dirname(model_path)) as fout:
        model.save(fout)
        os.link(fout.name, model_path)
