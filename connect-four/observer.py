import datetime
import json
import os
from typing import Tuple

from game import TwoPlayerGame
from player import Player, AlphaConnectPlayer
from state import State, FOUR, Color


class Observer(object):
    def notify_new_state(self, game: TwoPlayerGame, state: State):
        pass

    def notify_new_action(self, game: TwoPlayerGame, player: Player, action: Tuple[int, int]):
        pass

    def notify_end_game(self, game: TwoPlayerGame):
        pass


class ConsoleObserver(Observer):
    def __init__(self, show_state=True, show_action=True, show_end=True):
        self.show_state = show_state
        self.show_action = show_action
        self.show_end = show_end

    def notify_new_state(self, game, state: State):
        if self.show_state:
            print(state)

    def notify_new_action(self, game, player: Player, action: Tuple[int, int]):
        if self.show_action:
            print('\n%s (%s) plays %s' % (player, game.current_state.next_color, action))

    def notify_end_game(self, game: TwoPlayerGame):
        if self.show_end:
            winner = game.current_state.winner
            if winner is not None:
                print('The winner is: %s (%s)' % (game.players[winner], winner))
            else:
                print('The game is a draw')


class AlphaConnectSerializer(Observer):
    def notify_end_game(self, game: TwoPlayerGame):
        if self.is_self_play(game):
            self.save_game(game)

    def is_self_play(self, game):
        player1 = game.players[Color.WHITE]
        player2 = game.players[Color.BROWN]
        return isinstance(player1, AlphaConnectPlayer) and player1 is player2

    def save_game(self, game):
        data = self.serialize(game)
        # todo use model iteration
        path = 'data/0/{date:%Y%m%d_%H%M%S}.json'.format(date=datetime.datetime.now())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fout:
            json.dump(data, fout, indent=4)
        print('Written game to: %s' % path)

    @classmethod
    def serialize(cls, game):
        winner_color = game.current_state.winner
        first_color = game.state_history[0].next_color
        data = {
            'winner': winner_color.value,
            'starter': first_color.value,
            'actions': ''.join([hex(action[0] * FOUR + action[1])[2:] for action in game.action_history]),
            'policies': [{action.to_hex(): value for action, value in policy.items()}
                         for policy in game.players[Color.WHITE].policy_history]
        }
        return data
