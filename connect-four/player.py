import time
from abc import ABCMeta, abstractmethod
from operator import itemgetter
from random import choice
from typing import Union

from analyzer import player_value
from state import State, Color, FOUR
from tree import MiniMaxNode, MonteCarloNode


class Player(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.color = None  # type: Union[Color, None]

    def __str__(self):
        return '%s (%s)' % (self.name, self.color)

    def __repr__(self):
        return '%s(name=%s)' % (self.__class__.__name__, self.name)

    @abstractmethod
    def decide(self, state: State):
        pass

    def set_color(self, color: Color):
        self.color = color


class ConsolePlayer(Player):
    def decide(self, state: State):
        actions = list(sorted(state.allowed_actions))
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
    def decide(self, state: State):
        actions = state.allowed_actions
        return choice(list(actions))


class GreedyPlayer(Player):
    def decide(self, state: State):
        action_values = {}
        for action in state.allowed_actions:
            new_state = state.take_action(action)
            action_values[action] = player_value(new_state, self.color)
        _, max_value = max(action_values.items(), key=itemgetter(1))
        best_actions = [action for action, value in action_values.items() if value == max_value]
        random_best_action = choice(best_actions)
        return random_best_action


class MiniMaxPlayer(Player):
    def __init__(self, name: str = None, depth=2):
        super().__init__(name)
        self.expands = sum(((FOUR * FOUR) ** d for d in range(depth + 1)))

    def decide(self, state: State):
        root = MiniMaxNode(state, self.color)
        frontier = [root]
        for i in range(self.expands):
            if len(frontier) > 0:
                next_node = frontier.pop(0)
                frontier.extend(next_node.expand())

        action_values = {action: node.value for action, node in root.children.items()}
        _, max_value = max(action_values.items(), key=itemgetter(1))
        best_actions = [action for action, value in action_values.items() if value == max_value]
        random_best_action = choice(best_actions)
        return random_best_action


class MonteCarloPlayer(Player):
    def __init__(self, name: str, budget=1000):
        self.budget = budget
        super().__init__(name)

    def decide(self, state: State):
        t0 = time.time()
        root = MonteCarloNode(state, self.color)
        while time.time() - t0 < self.budget / 1000:
            root.search()
        return root.best_action()
