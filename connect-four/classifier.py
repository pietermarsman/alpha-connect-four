import json
import os
from random import sample

import numpy as np
from keras import Input, Model, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, AveragePooling3D, MaxPooling3D, Maximum, Reshape, \
    RepeatVector, Permute, BatchNormalization, Activation, Add
from keras.optimizers import Adam

from state import State, FOUR, Action, Color

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def fit_model(n_games=1000, samples_per_game=5, fit=True):
    output_path = new_model_path()
    input_shape = State.empty().to_numpy().shape[-1]
    model = create_model(input_shape, 10)
    print(model.summary())

    if fit:
        x_state, y_policy, y_reward = read_data(n_games, samples_per_game)
        model.fit(x_state, [y_policy, y_reward], epochs=100, validation_split=0.3,
                  callbacks=[EarlyStopping(patience=2)])

    model.save(output_path)


def new_model_path():
    files = os.listdir(MODEL_DIR)
    model_files = [f for f in files if f.endswith('h5')]
    model_iteration = len(model_files)
    output_path = os.path.abspath(os.path.join(MODEL_DIR, '%6.6d.h5' % model_iteration))
    return output_path


def read_data(n_games, n_samples_per_game):
    files = os.listdir(DATA_DIR)
    game_names = list(sorted([f for f in files if f.endswith('.json')]))
    game_names = game_names[-n_games:]

    states = []
    actions = []
    policies = []
    rewards = []

    for game_name in game_names:
        game_path = os.path.join(DATA_DIR, game_name)
        with open(game_path, 'r') as fin:
            game = json.load(fin)

        state = State.empty()
        game_states = []
        game_states.append(state)

        for action_hex, sparse_policy in zip(game['actions'], game['policies']):
            action = Action.from_hex(action_hex)
            actions.append(action)
            state = state.take_action(action)
            policy = [sparse_policy.get(action.to_hex(), 0.0) for action in Action.iter_actions()]
            policies.append(policy)
            game_states.append(state)

        for game_state in game_states[:-1]:
            states.append(game_state)
            rewards.append(_encode_winner(state.winner, game_state))

    # todo sample 8 states per game and augment
    data = sample(list(zip(states, policies, rewards)), len(game_names) * n_samples_per_game)
    x = []
    y_policy = []
    y_reward = []
    for state, policy, reward in data:
        x.append(state.to_numpy())
        y_policy.append(policy)
        y_reward.append(reward)

    return np.array(x), np.array(y_policy), np.array(y_reward)


def _encode_winner(winner: Color, state: State):
    if winner is state.next_color:
        return 1.0
    elif winner is state.next_color.other():
        return -1.0
    else:
        return 0.0


def create_model(input_size, kernel_size, c=10 ** -4):
    l2 = regularizers.l2(c)
    input = Input(shape=(FOUR, FOUR, FOUR, input_size))
    input_conv = Conv3D(kernel_size, 1, kernel_regularizer=l2)(input)
    pool1 = connect_layer(input_conv, kernel_size, l2)
    pool2 = connect_layer(pool1, kernel_size, l2)
    pool3 = connect_layer(pool2, kernel_size, l2)
    pool4 = connect_layer(pool3, kernel_size, l2)
    pool5 = connect_layer(pool4, kernel_size, l2)
    collapse = MaxPooling3D((1, 1, 4), 1)(pool5)
    flatten = Flatten()(collapse)
    # todo add some layers for the two output heads
    output_play = Dense(16, activation='softmax')(flatten)
    output_win = Dense(1, activation='tanh')(flatten)

    model = Model(inputs=input, outputs=[output_play, output_win])
    optimizer = Adam()
    metics = {'dense_1': 'categorical_accuracy', 'dense_2': 'mae'}
    model.compile(optimizer, ['categorical_crossentropy', 'mse'], metrics=metics)
    return model


def connect_layer(input, kernel_size, l2):
    """Residual layer modeled after AlphaGo"""
    pool1 = line_convolution(input, kernel_size, l2)
    norm1 = BatchNormalization()(pool1)
    relu1 = Activation('relu')(norm1)
    pool2 = line_convolution(relu1, kernel_size, l2)
    norm2 = BatchNormalization()(pool2)
    add = Add()([input, norm2])
    relu2 = Activation('relu')(add)

    return relu2


def line_convolution(input, kernel_size, l2):
    # todo add diagonal connection
    conv1 = Conv3D(kernel_size, 1, kernel_regularizer=l2)(input)
    permute_x1 = pool_direction(conv1, kernel_size, 0)
    permute_y1 = pool_direction(conv1, kernel_size, 1)
    permute_z1 = pool_direction(conv1, kernel_size, 2)
    pool1 = Maximum()([permute_x1, permute_y1, permute_z1])
    return pool1


def pool_direction(conv, kernel_size, direction):
    pool_size = [1, 1, 1]
    pool_size[direction] = FOUR

    permute_dims = [1, 2, 3, 4]
    permute_dims.insert(0, permute_dims.pop(direction))

    pool = AveragePooling3D(pool_size, 1)(conv)
    gather = Reshape((FOUR * FOUR * kernel_size,))(pool)
    repeat = RepeatVector(FOUR)(gather)
    spread = Reshape((FOUR, FOUR, FOUR, kernel_size))(repeat)
    permute = Permute(permute_dims)(spread)
    return permute


if __name__ == '__main__':
    # todo move to argparse in __main__.py
    fit_model(fit=False)
