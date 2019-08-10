import numpy as np
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv3D, Flatten, AveragePooling3D, MaxPooling3D, Maximum, Reshape, \
    RepeatVector, Permute
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from tqdm import tqdm

from game import TwoPlayerGame
from player import RandomPlayer
from state import State, FOUR, Stone


def generate_data(n_games):
    global dataset, labels, game
    dataset = []
    labels = []
    for i in tqdm(range(n_games)):
        board = State()
        player1 = RandomPlayer()
        player2 = RandomPlayer()

        game = TwoPlayerGame(board, player2, player1)
        game.play()

        if game.current_state.has_winner():
            history_size = len(game.state_history)
            dataset.append(to_numpy(game.state_history[history_size - 2]))
            labels.append(bool(game.current_state.winner().value - 1))

    dataset = np.stack(dataset)
    labels = np.array(labels)

    return dataset, labels


def create_model(kernel_size):
    # todo batch normalization
    # todo l2 regularization
    input = Input(shape=(FOUR, FOUR, FOUR, kernel_size))
    pool1 = connect_layer(input, kernel_size)
    pool2 = connect_layer(pool1, kernel_size)
    pool3 = connect_layer(pool2, kernel_size)
    pool4 = connect_layer(pool3, kernel_size)
    pool5 = connect_layer(pool4, kernel_size)
    collapse = MaxPooling3D((1, 1, 4), 1)(pool5)
    flatten = Flatten()(collapse)
    # todo predict move and winner (-1, 1)
    output = Dense(1, activation='sigmoid')(flatten)

    model = Model(inputs=input, outputs=output)
    optimizer = Adam()
    model.compile(optimizer, 'binary_crossentropy', metrics=[binary_accuracy])
    return model


def connect_layer(input, kernel_size):
    # todo residual layer
    conv = Conv3D(kernel_size, 1)(input)

    permute_x = pool_direction(conv, kernel_size, 0)
    permute_y = pool_direction(conv, kernel_size, 1)
    permute_z = pool_direction(conv, kernel_size, 2)

    pool = Maximum()([permute_x, permute_y, permute_z])
    return pool


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
    arr = [[[_encode_position(board, (x, y, z)) for z in range(FOUR)] for y in range(FOUR)] for x in range(FOUR)]
    return np.array(arr)


def _encode_position(state, pos):
    x, y, z = pos

    stone = state[pos]
    reachable = z == state._height(x, y)

    corner = (x == 0 or x == 3) and (y == 0 or y == 3)
    side = (x == 0 or x == 3 or y == 0 or y == 3) and not corner
    middle = not (corner or side)
    bottom = (z == 0)
    top = (z == 3)
    middle_z = not (bottom or top)
    center = middle and middle_z

    return (
        stone == Stone.NONE,
        stone == Stone.BROWN,
        stone == Stone.WHITE,
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
    dataset, labels = generate_data(100)

    print('Dataset shape:', dataset.shape)
    print('Labels shape:', labels.shape)

    model = create_model(11)
    print(model.summary())
    model.fit(dataset, labels, epochs=100, validation_split=.3, callbacks=[EarlyStopping(patience=2)])
