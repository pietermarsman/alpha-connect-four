import datetime
import json
import os
from typing import Tuple, List, Dict

from game import TwoPlayerGame
from player import Player, AlphaConnectPlayer
from state import State, FOUR, Color, Action


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
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def notify_end_game(self, game: TwoPlayerGame):
        if self.is_self_play(game):
            self.save_game(game)

    def is_self_play(self, game):
        player1 = game.players[Color.WHITE]
        player2 = game.players[Color.BROWN]
        return isinstance(player1, AlphaConnectPlayer) and player1 is player2

    def save_game(self, game):
        data = self.serializer(game)
        os.makedirs(os.path.dirname(self.data_dir), exist_ok=True)
        path = os.path.join(self.data_dir, '{date:%Y%m%d_%H%M%S_%f}.json'.format(date=datetime.datetime.now()))
        with open(path, 'w') as fout:
            json.dump(data, fout)
        print('Written game to: %s' % path)

    @staticmethod
    def serializer(game: TwoPlayerGame) -> Dict:
        winner_color = game.current_state.winner.value
        first_color = game.state_history[0].next_color.value
        actions = ''.join([hex(action[0] * FOUR + action[1])[2:] for action in game.action_history])
        policy_history = game.players[Color.WHITE].policy_history
        policies = [{action.to_hex(): value for action, value in policy.items()} for policy in policy_history]
        return {'winner': winner_color, 'starter': first_color, 'actions': actions, 'policies': policies}

    @staticmethod
    def deserialize(data) -> Tuple[Color, Color, List[Action], List[Dict[Action, float]]]:
        winner = Color(data['winner'])
        starter = Color(data['starter'])
        actions = [Action.from_hex(action_hex) for action_hex in data['actions']]
        policies = [{Action.from_hex(action_hex): value for action_hex, value in policy.items()}
                    for policy in data['policies']]
        return winner, starter, actions, policies
