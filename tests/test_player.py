import os

import pytest
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.optimizers import Adam

from game import TwoPlayerGame
from player import AlphaConnectPlayer
from state import State, FOUR


@pytest.fixture
def test_model_path():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_model.h5'))

    if not os.path.exists(model_path):
        kernel_size = 9
        input = Input(shape=(FOUR, FOUR, FOUR, kernel_size))
        flatten = Flatten()(input)
        output_play = Dense(16, activation='softmax')(flatten)
        output_win = Dense(1, activation='tanh')(flatten)

        model = Model(inputs=input, outputs=[output_play, output_win])
        optimizer = Adam()
        model.compile(optimizer, 'mse')
        model.save(model_path)

    return model_path


def test_alpha_connect_player_saves_policy(test_model_path):
    player = AlphaConnectPlayer('alpha', test_model_path, budget=5)
    game = TwoPlayerGame(State.empty(), player, player)
    assert len(game.action_history) == len(player.policy_history)
