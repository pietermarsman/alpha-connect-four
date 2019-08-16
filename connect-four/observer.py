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
            print('The winner is: %s (%s)' % (game.players[winner], winner))


class GameSerializer(Observer):
    def __init__(self):
        self.path = 'data/{date:%Y%m%d_%H%M%S}.json'.format(date=datetime.datetime.now())

    def notify_end_game(self, game: TwoPlayerGame):
        if game.datetime_end is not None:
            self.save_game(game)

    def save_game(self, game):
        data = self.serialize(game)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as fout:
            json.dump(data, fout, indent=4)

    @classmethod
    def serialize(cls, game):
        winner_color = game.current_state.winner
        first_color = game.state_history[0].next_color
        data = {
            'winner': winner_color.value,
            'starter': first_color.value,
            'game': ''.join([hex(action[0] * FOUR + action[1])[2:] for action in game.action_history])
        }
        return data


class AlphaConnectSerializer(GameSerializer):
    @classmethod
    def serialize(cls, game):
        data = super().serialize(game)

        player1 = game.players[Color.WHITE]
        player2 = game.players[Color.BROWN]
        is_self_play = isinstance(player1, AlphaConnectPlayer) and player1 is player2
        if is_self_play:
            data['policies'] = [{action.to_hex(): value for action, value in policy.items()}
                                for policy in player1.policy_history]
        return data
