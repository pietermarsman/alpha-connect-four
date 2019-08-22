import json
from random import sample

import numpy as np
from tensorflow.python.keras import Input, Model, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.python.keras.layers import Dense, Conv3D, Flatten, Reshape, \
    RepeatVector, Permute, BatchNormalization, Concatenate, ReLU
from tensorflow.python.keras.optimizers import Adam
from tqdm import tqdm

from observer import AlphaConnectSerializer
from state import State, FOUR, Action, Color, Augmentation
from util import list_files


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
    game_files = list_files(data_path, '.json')

    if max_games is not None:
        game_files = list(game_files)[-max_games:]

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

        n_samples = min(len(states), 8)
        game_samples = sample(list(range(len(states))), n_samples)
        for augmentation, i in zip(Augmentation.iter_augmentations(), game_samples):
            augmentend_action_order = sorted(Action.iter_actions(), key=lambda a: a.augment(augmentation).to_int())

            x.append(states[i].to_numpy(augmentation))
            y_policy.append([policies[i].get(action, 0.0) for action in augmentend_action_order])
            y_reward.append(_encode_winner(final_state.winner, states[i]))

    return np.array(x), np.array(y_policy), np.array(y_reward)


def _encode_winner(winner: Color, state: State):
    if winner is state.next_color:
        return 1.0
    elif winner is state.next_color.other():
        return -1.0
    else:
        return 0.0


def create_model(input_size, filters, c=10 ** -4, initialization='he_normal'):
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
    output_play = Dense(16, activation='softmax')(dense_play)

    collapse_win = normalized_relu(Conv3D(3, (1, 1, 4), kernel_regularizer=l2)(last_conv))
    flatten_win = Flatten()(collapse_win)
    dense_win = Dense(16, activation='relu', kernel_regularizer=l2)(flatten_win)
    output_win = Dense(1, activation='tanh', kernel_regularizer=l2)(dense_win)

    model = Model(inputs=input, outputs=[output_play, output_win])
    optimizer = Adam()
    metics = {'dense_1': 'categorical_accuracy', 'dense_3': 'mae'}
    model.compile(optimizer, ['categorical_crossentropy', 'mse'], metrics=metics)
    return model


def line_convolution(input, filters, l2):
    conv = Conv3D(filters, 1, kernel_regularizer=l2)(input)
    # todo use same layer for x and y direction
    permute_x1 = axis_convolution(conv, filters // 3, 0, l2)
    permute_y1 = axis_convolution(conv, filters // 3, 1, l2)
    permute_z1 = axis_convolution(conv, filters // 3, 2, l2)
    permute_xy = plane_convolution(conv, filters // 4, 0, 1, l2)
    permute_xz = plane_convolution(conv, filters // 4, 0, 2, l2)
    permute_yz = plane_convolution(conv, filters // 4, 1, 2, l2)
    permute_xyz = box_convolution(conv, filters // 6, l2)
    concatenate = Concatenate()(
        [conv, permute_x1, permute_y1, permute_z1, permute_xy, permute_xz, permute_yz, permute_xyz])
    return concatenate


def axis_convolution(input, filters, direction, l2):
    kernel_size = [1, 1, 1]
    kernel_size[direction] = FOUR

    permute_dims = [1, 2, 3, 4]
    permute_dims.insert(0, permute_dims.pop(direction))

    reduce = Conv3D(filters, kernel_size, kernel_regularizer=l2)(input)
    gather = Reshape((FOUR * FOUR * filters,))(reduce)
    repeat = RepeatVector(FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, filters))(repeat)
    permute = Permute(permute_dims)(spread)
    return permute


def plane_convolution(input, filters, direction1, direction2, l2):
    kernel_size = [1, 1, 1]
    kernel_size[direction1] = FOUR
    kernel_size[direction2] = FOUR

    permute_dims = [1, 2, 3, 4]
    permute_dims.insert(0, permute_dims.pop(direction2))
    permute_dims.insert(0, permute_dims.pop(direction1))

    reduce = Conv3D(filters, 1, kernel_regularizer=l2)(input)
    squash = Conv3D(filters, kernel_size, kernel_regularizer=l2)(reduce)
    gather = Reshape((FOUR * filters,))(squash)
    repeat = RepeatVector(FOUR * FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, filters))(repeat)
    permute = Permute(permute_dims)(spread)
    return permute


def box_convolution(input, filters, l2):
    reduce = Conv3D(filters, 1, kernel_regularizer=l2)(input)
    squash = Conv3D(filters, (FOUR, FOUR, FOUR), kernel_regularizer=l2)(reduce)
    gather = Reshape((filters,))(squash)
    repeat = RepeatVector(FOUR * FOUR * FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, filters))(repeat)
    return spread


def normalized_relu(layer):
    norm = BatchNormalization()(layer)
    relu = ReLU()(norm)
    return relu
