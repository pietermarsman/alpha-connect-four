from abc import ABCMeta, abstractmethod
from random import choice

from state import ConnectFour3D


class Player(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    @abstractmethod
    def decide(self, state: ConnectFour3D):
        pass


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
