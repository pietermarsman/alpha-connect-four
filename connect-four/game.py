import datetime
import json
import os
from uuid import uuid4

from player import Player
from state import ConnectFour3D, Stone, FOUR


class TwoPlayerGame(object):
    """Two player game orchestrator"""

    def __init__(self, state: ConnectFour3D, player1: Player, player2: Player, observers: list=None):
        self.current_state = state
        player1.set_color(Stone.WHITE)
        player2.set_color(Stone.BROWN)
        self.players = {player.color: player for player in [player1, player2]}

        if observers is None:
            self.observers = []
        else:
            self.observers = observers

        self.game_serializer = GameSerializer()
        self.state_history = [self.current_state]
        self.action_history = []
        self.datetime_start = None
        self.datetime_end = None

    def play(self):
        self.datetime_start = datetime.datetime.utcnow()
        while not self.current_state.is_end_of_game():
            self._turn()
        self.datetime_end = datetime.datetime.utcnow()
        self.announce_winner()
        self.save_game()

    def _turn(self):
        player = self.next_player()
        pos = player.decide(self.current_state)
        self.current_state = self.current_state.take_action(pos)

        self.action_history.append(pos)
        self.state_history.append(self.current_state)

        self._notify(player, pos, self.current_state)

    def next_player(self) -> Player:
        return self.players[self.current_state.next_stone]

    def _notify(self, player, action, new_state):
        for observer in self.observers:
            observer.notify_new_action(player, action)
            observer.notify_new_state(new_state)

    def announce_winner(self):
        winner = self.current_state.winner()
        if winner is not None:
            print('The winner is: %s' % self.players[winner])

    def save_game(self):
        path = 'data/{date:%Y%m%d}/{id}.json'.format(date=datetime.datetime.now(), id=uuid4())
        self.game_serializer.save_game(self, path)


class GameSerializer(object):
    @staticmethod
    def serialize(game):
        winner_stone = game.current_state.winner()
        first_stone = game.state_history[0].next_stone
        players = {stone: {'hash': hash(player), 'implementation': repr(player)}
                   for stone, player in game.players.items()}

        return {
            'meta': {
                'start': game.datetime_start.isoformat(),
                'end': game.datetime_end.isoformat(),
                'duration': (game.datetime_end - game.datetime_start).total_seconds()
            },
            'players': list(players.values()),
            'winner': players[winner_stone],
            'start player': players[first_stone],
            'game': [hex(action[0] * FOUR + action[1]) for action in game.action_history]
        }

    @staticmethod
    def save_game(game, path):
        data = GameSerializer.serialize(game)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fout:
            json.dump(data, fout, indent=4)
