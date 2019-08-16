from game import TwoPlayerGame
from player import AlphaConnectPlayer
from state import State


def test_alpha_connect_player_saves_policy():
    player = AlphaConnectPlayer('alpha', '', budget=5)
    game = TwoPlayerGame(State.empty(), player, player)
    assert len(game.action_history) == len(player.policy_history)
