import time
from abc import ABCMeta, abstractmethod
from operator import itemgetter
from random import choice

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.saving import load_model

from analyzer import player_value
from state import State, FOUR, Action
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
        while True:
            print('Possible actions:', ', '.join(map(str, sorted(state.allowed_actions))))
            user_input = input('Choose your action: ')

            try:
                action = Action.from_hex(user_input)
                if action in state.allowed_actions:
                    return action

            except ValueError:
                pass


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
    def __init__(self, name: str, model_path, exploration=1.0, start_temperature=1.0, time_budget=None,
                 search_budget=None):
        self.model = self.load_model(model_path)
        self.root = AlphaConnectNode(State.empty(), self.model)
        self.exploration = exploration
        self._temperature = start_temperature

        if time_budget is not None and search_budget is None:
            self.budget_type = 'time'
            self.budget = time_budget
        elif time_budget is None and search_budget is not None:
            self.budget_type = 'search'
            self.budget = search_budget
        else:
            raise ValueError('Either time_budget xor search_budget should be None, but not both or neither')

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
        if self.root is None:
            self.root = AlphaConnectNode(state, self.model)
        self.root.parent = None

        if self.budget_type == 'time':
            while time.time() - t0 < self.budget / 1000:
                self.root.search(self.exploration)
        else:
            for _ in range(self.budget):
                self.root.search(self.exploration)

        self.save_policy()
        action = self.root.sample_action(self.temperature(state))
        return action

    def temperature(self, state: State):
        """AlphaGo lowers the temperature to infinitesimal after 30 moves

        Connect Four is a smaller game, so we use 16 moves
        """
        if state.number_of_stones < 16:
            return self._temperature
        else:
            return 0.1

    def save_policy(self):
        temperature = self.temperature(self.root.state)
        self.policy_history.append(self.root.policy(temperature))
