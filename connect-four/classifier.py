import numpy as np
from keras import Input, Model, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, AveragePooling3D, MaxPooling3D, Maximum, Reshape, \
    RepeatVector, Permute, BatchNormalization, Activation, Add
from keras.optimizers import Adam
from tqdm import tqdm

from game import TwoPlayerGame
from player import RandomPlayer
from state import State, FOUR, Color, Action, Position


def generate_data(n_games):
    global dataset, labels, game
    dataset = []
    labels = []
    actions = []
    for _ in tqdm(range(n_games)):
        board = State.empty()
        player1 = RandomPlayer()
        player2 = RandomPlayer()

        game = TwoPlayerGame(board, player2, player1)
        game.play()

        history_index = len(game.state_history) - 2
        state = game.state_history[history_index]
        dataset.append(to_numpy(state))
        winner = game.current_state.winner
        if winner == state.next_color:
            labels.append(1)
        elif winner == state.next_color.other():
            labels.append(-1)
        else:
            labels.append(0)
        actions.append(game.action_history[history_index].to_int())

    dataset = np.stack(dataset)
    labels = np.array(labels)
    action_array = np.zeros((len(actions), FOUR * FOUR))
    action_array[np.arange(action_array.shape[0]), actions] = 1

    return dataset, labels, action_array


def create_model(kernel_size):
    input = Input(shape=(FOUR, FOUR, FOUR, kernel_size))
    pool1 = connect_layer(input, kernel_size)
    pool2 = connect_layer(pool1, kernel_size)
    pool3 = connect_layer(pool2, kernel_size)
    pool4 = connect_layer(pool3, kernel_size)
    pool5 = connect_layer(pool4, kernel_size)
    collapse = MaxPooling3D((1, 1, 4), 1)(pool5)
    flatten = Flatten()(collapse)
    output_play = Dense(16, activation='softmax')(flatten)
    output_win = Dense(1, activation='tanh')(flatten)

    model = Model(inputs=input, outputs=[output_play, output_win])
    optimizer = Adam()
    metics = {'dense_1': 'categorical_accuracy', 'dense_2': 'mae'}
    model.compile(optimizer, ['categorical_crossentropy', 'mse'], metrics=metics)
    return model


def connect_layer(input, kernel_size, c=10 ** -4):
    """Residual layer modeled after AlphaGo"""
    l2 = regularizers.l2(c)
    pool1 = line_convolution(input, kernel_size, l2)
    norm1 = BatchNormalization()(pool1)
    relu1 = Activation('relu')(norm1)
    pool2 = line_convolution(relu1, kernel_size, l2)
    norm2 = BatchNormalization()(pool2)
    add = Add()([input, norm2])
    relu2 = Activation('relu')(add)

    return relu2


def line_convolution(input, kernel_size, l2):
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


def to_numpy(board):
    arr = [[[_encode_position(board, Position(x, y, z)) for z in range(FOUR)] for y in range(FOUR)] for x in
           range(FOUR)]
    return np.array(arr)


def _encode_position(state: State, pos: Action):
    x, y, z = pos

    stone = state.stones[pos]
    reachable = z == state.pin_height[Action(x, y)]

    corner = (x == 0 or x == 3) and (y == 0 or y == 3)
    side = (x == 0 or x == 3 or y == 0 or y == 3) and not corner
    middle = not (corner or side)
    bottom = (z == 0)
    top = (z == 3)
    middle_z = not (bottom or top)
    center = middle and middle_z

    return (
        stone == Color.NONE,
        stone == Color.BROWN,
        stone == Color.WHITE,
        reachable,
        corner,
        side,
        middle,
        bottom,
        top,
        middle_z,
        center
    )


if __name__ == '__main__':
    dataset, labels, actions = generate_data(10000)

    print('Dataset shape:', dataset.shape)
    print('Labels shape:', labels.shape)

    model = create_model(11)
    print(model.summary())
    model.fit(dataset, [actions, labels], epochs=100, validation_split=.3, callbacks=[EarlyStopping(patience=2)])
