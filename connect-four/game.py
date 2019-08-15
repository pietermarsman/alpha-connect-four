import datetime
from typing import Dict

from player import Player
from state import State, Color


class TwoPlayerGame(object):
    """Two player game orchestrator"""

    def __init__(self, state: State, player1: Player, player2: Player, observers: list = None):
        self.current_state = state  # type: State
        player1.set_color(Color.WHITE)
        player2.set_color(Color.BROWN)
        self.players = {player.color: player for player in [player1, player2]}  # type: Dict[Color, Player]

        if observers is None:
            self.observers = []
        else:
            self.observers = observers

        self.state_history = [self.current_state]
        self.action_history = []
        self.datetime_start = None
        self.datetime_end = None

    def play(self):
        self.datetime_start = datetime.datetime.utcnow()
        while not self.current_state.is_end_of_game():
            self._turn()
        self.datetime_end = datetime.datetime.utcnow()
        self._notify_end_game()

    def _turn(self):
        player = self.next_player()
        pos = player.decide(self.current_state)
        self._notify_action(player, pos)

        self.current_state = self.current_state.take_action(pos)
        self._notify_new_state(self.current_state)

        self.action_history.append(pos)
        self.state_history.append(self.current_state)

    def next_player(self) -> Player:
        return self.players[self.current_state.next_color]

    def _notify_action(self, player, action):
        for observer in self.observers:
            observer.notify_new_action(self, player, action)

    def _notify_new_state(self, new_state):
        for observer in self.observers:
            observer.notify_new_state(self, new_state)

    def _notify_end_game(self):
        for observer in self.observers:
            observer.notify_end_game(self)
