import os
from typing import List

import pytest
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Flatten, Dense
from tensorflow.python.keras.optimizers import Adam

from game import TwoPlayerGame
from player import AlphaConnectPlayer, GreedyPlayer, MiniMaxPlayer, MonteCarloPlayer, Player
from state import State, FOUR, Action


@pytest.fixture
def test_model_path():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'test_model.h5'))

    if not os.path.exists(model_path):
        filters = 13
        input = Input(shape=(FOUR, FOUR, FOUR, filters))
        flatten = Flatten()(input)
        output_play = Dense(16, activation='softmax')(flatten)
        output_win = Dense(1, activation='tanh')(flatten)

        model = Model(inputs=input, outputs=[output_play, output_win])
        optimizer = Adam()
        model.compile(optimizer, 'mse')
        model.save(model_path)

    return model_path


@pytest.fixture
def players(test_model_path):
    return [GreedyPlayer(), MiniMaxPlayer(), MonteCarloPlayer(),
            AlphaConnectPlayer(test_model_path, exploration=10.0, search_budget=16 * 16, self_play=True)]


@pytest.fixture
def win_in_one_move():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(3, 3))
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(3, 2))
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(3, 1))
    return state


@pytest.fixture
def other_win_in_one_move():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(3, 3))
    state = state.take_action(Action(1, 2))
    state = state.take_action(Action(3, 2))
    state = state.take_action(Action(2, 1))
    state = state.take_action(Action(3, 1))
    return state


def test_alpha_connect_player_saves_policy(test_model_path):
    player = AlphaConnectPlayer(test_model_path, 'alpha', time_budget=5)
    game = TwoPlayerGame(State.empty(), player, player)
    assert len(game.action_history) == len(player.history)


def test_player_wins_when_possible(players: List[Player], win_in_one_move: State):
    for player in players:
        action = player.decide(win_in_one_move)
        new_state = win_in_one_move.take_action(action)
        assert new_state.has_winner(), '%s does not play winning move' % player


def test_player_prevents_other_from_winning(players: List[Player], other_win_in_one_move: State):
    for player in players:
        action = player.decide(other_win_in_one_move)
        assert Action(3, 0) == action, '%s does not prevent other from winning' % player
