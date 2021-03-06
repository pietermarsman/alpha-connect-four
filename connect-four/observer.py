import datetime
import json
import os
from typing import Tuple, List, Dict

from game import TwoPlayerGame
from player import Player, AlphaConnectPlayer
from state import State, FOUR, Color, Action
from util import format_in_action_grid


class Observer(object):
    def notify_new_state(self, game: TwoPlayerGame, state: State):
        pass

    def notify_new_action(self, game: TwoPlayerGame, player: Player, action: Tuple[int, int]):
        pass

    def notify_end_game(self, game: TwoPlayerGame):
        pass


class GameStatePrinter(Observer):
    def __init__(self, show_state=True, show_action=True, show_winner=True, show_action_history=False):
        self.show_state = show_state
        self.show_action = show_action
        self.show_winner = show_winner
        self.show_action_history = show_action_history

    def notify_new_state(self, game, state: State):
        if self.show_state:
            print(state, end='\n\n')

    def notify_new_action(self, game, player: Player, action: Tuple[int, int]):
        if self.show_action:
            print('%s (%s) plays %s' % (player, game.current_state.next_color, action))
        if self.show_action_history:
            actions = ''.join([action.to_hex() for action in game.action_history])
            print('Action history: %s' % actions)

    def notify_end_game(self, game: TwoPlayerGame):
        if self.show_winner:
            winner = game.current_state.winner
            if winner is not None:
                print('The winner is: %s (%s)' % (game.players[winner], winner))
            else:
                print('The game is a draw')


class GameWinnerSerializer(Observer):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def notify_end_game(self, game: TwoPlayerGame):
        self.save_game(game)

    def save_game(self, game: TwoPlayerGame):
        data = self.serialize(game)
        path = os.path.join(self.data_dir, '{date:%Y%m%d_%H%M%S_%f}.json'.format(date=datetime.datetime.now()))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fout:
            json.dump(data, fout)
        print('Written game result to: %s' % path)

    @staticmethod
    def serialize(game: TwoPlayerGame):
        winner = game.players.get(game.current_state.winner, None)
        return {
            'white': repr(game.players[Color.WHITE]),
            'brown': repr(game.players[Color.BROWN]),
            'winner': repr(winner)
        }


class AlphaConnectSerializer(Observer):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def notify_end_game(self, game: TwoPlayerGame):
        if self.is_self_play(game) and game.current_state.has_winner():
            self.save_game(game)

    @staticmethod
    def is_self_play(game):
        player1 = game.players[Color.WHITE]
        player2 = game.players[Color.BROWN]
        return isinstance(player1, AlphaConnectPlayer) and player1 is player2

    def save_game(self, game):
        data = self.serializer(game)
        path = os.path.join(self.data_dir, '{date:%Y%m%d_%H%M%S_%f}.json'.format(date=datetime.datetime.now()))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fout:
            json.dump(data, fout)
        print('Written game to: %s' % path)

    @staticmethod
    def serializer(game: TwoPlayerGame) -> Dict:
        winner_color = game.current_state.winner.value
        first_color = game.state_history[0].next_color.value
        actions = ''.join([hex(action[0] * FOUR + action[1])[2:] for action in game.action_history])
        player_history = game.players[Color.WHITE].history
        policy_history = [hist['policy'] for hist in player_history]
        policies = [{action.to_hex(): value for action, value in policy.items()} for policy in policy_history]
        value_history = [hist['total_value'] / hist['visit_count'] for hist in player_history]
        return {'winner': winner_color, 'starter': first_color, 'actions': actions, 'policies': policies,
                'values': value_history}

    @staticmethod
    def deserialize(data) -> Tuple[Color, Color, List[Action], List[Dict[Action, float]]]:
        winner = Color(data['winner'])
        starter = Color(data['starter'])
        actions = [Action.from_hex(action_hex) for action_hex in data['actions']]
        policies = [{Action.from_hex(action_hex): value for action_hex, value in policy.items()}
                    for policy in data['policies']]
        return winner, starter, actions, policies


class AlphaConnectPrinter(Observer):
    def notify_new_action(self, game: TwoPlayerGame, player: Player, action: Tuple[int, int]):
        if isinstance(player, AlphaConnectPlayer):
            mcts_policy = player.root.policy(1.0)
            mcts_value = player.root.average_value
            mcts_emotion = self.express_evaluation_as_emotion(mcts_value)
            raw_policy, raw_value = self.raw_predictions(player, game.current_state)

            print('%s is done' % player)
            print('It looked at %d states' % player.root.visit_count)
            print('At first, it values the current state as %.2f' % raw_value)
            print('At first, it wants to play:\n%s' % format_in_action_grid(raw_policy))
            print('After searching, it values the current state as %.2f' % mcts_value)
            print('After searching, it wants to play:\n%s' % format_in_action_grid(mcts_policy))
            print('It feels %s' % mcts_emotion, end='\n\n')

    @staticmethod
    def express_evaluation_as_emotion(evaluation):
        if -1 < evaluation <= -.75:
            emotion = u'\U0001F62D'
        elif -.75 < evaluation <= -.25:
            emotion = u'\U0001F61F'
        elif -.25 < evaluation <= .25:
            emotion = u'\U0001F610'
        elif .25 < evaluation < .75:
            emotion = u'\U0001F642'
        else:
            emotion = u'\U0001F600'
        return emotion

    @staticmethod
    def raw_predictions(player: AlphaConnectPlayer, state: State):
        pred_actions, pred_value = player.model.model.predict(state.to_numpy(batch=True))
        action_probs = dict(zip(Action.iter_actions(), pred_actions[0]))
        return action_probs, pred_value[0].item()
