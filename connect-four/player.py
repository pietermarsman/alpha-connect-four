import time
from abc import ABCMeta, abstractmethod
from operator import itemgetter
from random import choice

import numpy as np
from keras import backend as K
from keras.engine.saving import load_model

from analyzer import player_value
from state import State, FOUR
from tree import MiniMaxNode, MonteCarloNode, AlphaConnectNode


class Player(metaclass=ABCMeta):
    def __init__(self, name: str = None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return '%s(name=%s)' % (self.__class__.__name__, self.name)

    @abstractmethod
    def decide(self, state: State):
        pass


class ConsolePlayer(Player):
    def decide(self, state: State):
        actions = list(sorted(state.allowed_actions))
        action = None

        while action is None:
            # todo use new action names
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
            action_values[action] = player_value(new_state, state.next_color)
        _, max_value = max(action_values.items(), key=itemgetter(1))
        best_actions = [action for action, value in action_values.items() if value == max_value]
        random_best_action = choice(best_actions)
        return random_best_action


class MiniMaxPlayer(Player):
    def __init__(self, name: str = None, depth=2):
        super().__init__(name)
        self.expands = sum(((FOUR * FOUR) ** d for d in range(depth + 1)))

    def decide(self, state: State):
        root = MiniMaxNode(state, state.next_color)
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
    def __init__(self, name: str, exploration, budget=1000):
        self.root = MonteCarloNode(State.empty(), exploration=exploration)
        self.exploration = exploration
        self.budget = budget
        super().__init__(name)

    def decide(self, state: State):
        t0 = time.time()
        self.root = self.root.find_state(state)
        self.root.parent = None
        if self.root is None:
            self.root = MonteCarloNode(state, exploration=self.exploration)
        while time.time() - t0 < self.budget / 1000:
            self.root.search()
        return self.root.best_action()


class AlphaConnectPlayer(Player):
    def __init__(self, name: str, model_path, exploration=1.0, temperature=1.0, budget=1000):
        self.model = self.load_model(model_path)
        self.root = AlphaConnectNode(State.empty(), self.model, c_puct=exploration, temperature=temperature)
        self.exploration = exploration
        self.temperature = temperature
        self.budget = budget
        self.policy_history = []
        super().__init__(name)

    @staticmethod
    def load_model(model_path):
        model = load_model(model_path)
        # first prediction takes more time
        model.predict(np.array([State.empty().to_numpy()]))
        return model

    def clear_session(self):
        K.clear_session()

    def decide(self, state: State):
        t0 = time.time()
        self.root = self.root.find_state(state)
        # todo add dirichlet noise to root action probabilities
        if self.root is None:
            self.root = AlphaConnectNode(state, self.model, c_puct=self.exploration, temperature=self.temperature)
        self.root.parent = None
        while time.time() - t0 < self.budget / 1000:
            self.root.search()
        self.save_policy()
        action = self.root.sample_action()
        return action

    def save_policy(self):
        self.policy_history.append(self.root.policy())
