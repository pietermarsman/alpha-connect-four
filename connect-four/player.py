from abc import ABCMeta, abstractmethod
from operator import itemgetter, sub
from random import choice

from state import ConnectFour3D, Stone


class Player(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.name = name
        self.color = None  # type: Stone

    def __str__(self):
        return '%s (%s)' % (self.name, self.color)

    @abstractmethod
    def decide(self, state: ConnectFour3D):
        pass

    def set_color(self, color: Stone):
        self.color = color


class ConsolePlayer(Player):
    def decide(self, state: ConnectFour3D):
        actions = list(sorted(state.possible_actions()))
        action = None

        while action is None:
            print('Possible actions:')
            for i, action in enumerate(actions):
                print('%d. %s' % (i, action))
            user_input = input('Choose your action: ')

            try:
                user_choice = int(user_input)
                if user_choice < 0 or user_choice >= len(actions):
                    raise ValueError
                action = actions[user_choice]
            except ValueError:
                pass

        return action


class RandomPlayer(Player):
    def decide(self, state: ConnectFour3D):
        actions = state.possible_actions()
        return choice(list(actions))


class GreedyPlayer(Player):
    def decide(self, state: ConnectFour3D):
        action_values = {}
        for action in state.possible_actions():
            new_state = state.take_action(action)
            my_value = self.value(new_state, self.color)
            other_value = self.value(new_state, self.color.other())
            action_values[action] = tuple(map(sub, my_value, other_value))
        best_action, _ = max(action_values.items(), key=itemgetter(1))
        return best_action

    @staticmethod
    def value(state: ConnectFour3D, player_stone: Stone):
        value = [0, 0, 0, 0, 0]
        for connected_stones in GreedyPlayer.connected_stones_owned_by(state, player_stone):
            n_player_stones = sum([stone is player_stone for stone in connected_stones])
            value[4 - n_player_stones] += 1
        return tuple(value)


    @staticmethod
    def connected_stones_owned_by(state: ConnectFour3D, player_stone: Stone):
        for connected_stones in state.connected_stones():
            not_any_other = all((stone != player_stone.other() for stone in connected_stones))
            if not_any_other:
                yield connected_stones
