import datetime
import json
import os
from typing import Tuple
from uuid import uuid4

from game import TwoPlayerGame
from player import Player
from state import ConnectFour3D, FOUR


class Observer(object):
    def notify_new_state(self, game: TwoPlayerGame, state: ConnectFour3D):
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

    def notify_new_state(self, game, state: ConnectFour3D):
        if self.show_state:
            print(state)

    def notify_new_action(self, game, player: Player, action: Tuple[int, int]):
        if self.show_action:
            print('\n%s plays %s' % (player, action))

    def notify_end_game(self, game: TwoPlayerGame):
        if self.show_end:
            winner = game.current_state.winner()
            print('The winner is: %s' % game.players[winner])


class GameSerializer(Observer):
    def __init__(self):
        self.path = 'data/{date:%Y%m%d}/{id}.json'.format(date=datetime.datetime.now(), id=uuid4())

    def notify_end_game(self, game: TwoPlayerGame):
        if game.datetime_end is not None:
            self.save_game(game)

    def save_game(self, game):
        data = GameSerializer.serialize(game)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, 'w') as fout:
            json.dump(data, fout, indent=4)

    @staticmethod
    def serialize(game):
        winner_stone = game.current_state.winner()
        first_stone = game.state_history[0].next_stone
        players = {stone: {'hash': hash(player), 'implementation': repr(player)}
                   for stone, player in game.players.items()}
        data = {
            'meta': {
                'start': game.datetime_start.isoformat(),
                'end': game.datetime_end.isoformat(),
                'duration': (game.datetime_end - game.datetime_start).total_seconds()
            },
            'players': list(players.values()),
            'winner': players[winner_stone],
            'start player': players[first_stone],
            'game': ''.join([hex(action[0] * FOUR + action[1])[2:] for action in game.action_history])
        }
        return data
