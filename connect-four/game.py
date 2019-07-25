from player import Player
from state import ConnectFour3D, Stone


class TwoPlayerGame(object):
    """Two player game orchestrator"""

    def __init__(self, state: ConnectFour3D, player1: Player, player2: Player, observers: list=None):
        self.current_state = state
        self.state_history = [self.current_state]
        self.action_history = []
        player1.set_color(Stone.WHITE)
        player2.set_color(Stone.BROWN)
        self.players = {player.color: player for player in [player1, player2]}

        if observers is None:
            self.observers = []
        else:
            self.observers = observers

    def play(self):
        while not self.current_state.is_end_of_game():
            self._turn()
        self.announce_winner()

    def announce_winner(self):
        winner = self.current_state.winner()
        if winner is not None:
            print('The winner is: %s' % self.players[winner])

    def next_player(self) -> Player:
        return self.players[self.current_state.next_stone]

    def _turn(self):
        player = self.next_player()
        pos = player.decide(self.current_state)
        self.current_state = self.current_state.take_action(pos)

        self.action_history.append(pos)
        self.state_history.append(self.current_state)

        self._notify(player, pos, self.current_state)

    def _notify(self, player, action, new_state):
        for observer in self.observers:
            observer.notify_new_action(player, action)
            observer.notify_new_state(new_state)
