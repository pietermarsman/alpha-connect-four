from abc import ABCMeta, abstractmethod
from operator import itemgetter
from random import choice

from analyzer import player_value
from state import State, Stone


class Player(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name
        self.color = None  # type: Stone

    def __str__(self):
        return '%s (%s)' % (self.name, self.color)

    def __repr__(self):
        return '%s(name=%s)' % (self.__class__.__name__, self.name)

    @abstractmethod
    def decide(self, state: State):
        pass

    def set_color(self, color: Stone):
        self.color = color


class ConsolePlayer(Player):
    def decide(self, state: State):
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
    def decide(self, state: State):
        actions = state.possible_actions()
        return choice(list(actions))


class GreedyPlayer(Player):
    def decide(self, state: State):
        action_values = {}
        for action in state.possible_actions():
            new_state = state.take_action(action)
            action_values[action] = player_value(new_state, self.color)
        _, max_value = max(action_values.items(), key=itemgetter(1))
        best_actions = [action for action, value in action_values.items() if value == max_value]
        random_best_action = choice(best_actions)
        return random_best_action


class MiniMaxNode(object):
    def __init__(self, state, player_color, state_color=None, parent=None):
        self.state = state  # type: State
        self.player_color = player_color  # type: Stone
        if state_color is None:
            self.state_color = player_color
        else:
            self.state_color = state_color
        self.parent = parent
        self.children = None
        self.value = player_value(self.state, self.player_color)

    def expand(self):
        self.children = {
            action: MiniMaxNode(self.state.take_action(action), self.player_color, self.state_color.other(), self)
            for action in self.state.possible_actions()}
        self.propagate_value()
        return self.children.values()

    def propagate_value(self):
        if not self.state.is_end_of_game():
            if self.player_color is self.state_color:
                self.value = max([child.value for child in self.children.values()])
            else:
                self.value = min([child.value for child in self.children.values()])
        if self.parent is not None:
            self.parent.propagate_value()

    def __lt__(self, other):
        return self.value < other.value


class MiniMaxPlayer(Player):
    def __init__(self, name: str = None):
        super().__init__(name)
        self.expands = 16 ** 2

    def decide(self, state: State):
        root = MiniMaxNode(state, self.color)
        frontier = [root]
        for _ in range(self.expands):
            next_node = frontier.pop(0)
            frontier.extend(next_node.expand())

        action_values = {action: node.value for action, node in root.children.items()}
        _, max_value = max(action_values.items(), key=itemgetter(1))
        best_actions = [action for action, value in action_values.items() if value == max_value]
        random_best_action = choice(best_actions)
        return random_best_action
