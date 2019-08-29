import time
from abc import ABCMeta, abstractmethod
from operator import itemgetter
from random import choice
from typing import Union

import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.saving import load_model

from analyzer import player_value
from state import State, FOUR, Action
from tree import MiniMaxNode, MonteCarloNode, AlphaConnectNode, BatchEvaluator
from util import format_in_action_grid


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
            print('Possible actions:')
            print(format_in_action_grid({action: str(action) for action in Action.iter_actions()},
                                        cell_format='{:.2s}', default_value='  '))
            user_input = input('Choose your action: ')

            try:
                action = Action.from_hex(user_input)
                if action in state.allowed_actions:
                    print()
                    return action
                else:
                    print('Action %s not allowed' % action)

            except ValueError:
                print('User input is not an action')


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
    def __init__(self, name: str = None, exploration=1.0, budget=1000):
        self.root = MonteCarloNode(State.empty(), exploration=exploration)
        self.exploration = exploration
        self.budget = budget
        super().__init__(name)

    def decide(self, state: State):
        t0 = time.time()
        self.root = self.root.find_state(state)
        if self.root is None:
            self.root = MonteCarloNode(state, exploration=self.exploration)
        self.root.parent = None
        while time.time() - t0 < self.budget / 1000:
            self.root.search()
        return self.root.best_action()


class AlphaConnectPlayer(Player):
    def __init__(self, model_path, name: str = None, exploration=1.0, start_temperature=1.0, time_budget=None,
                 search_budget=None, self_play=False, batch_size=16):
        self.model = self.load_model(model_path, batch_size)
        self.exploration = exploration
        self._temperature = start_temperature
        self.is_self_play = self_play
        self.root = None  # type: Union[None, AlphaConnectNode]
        self.set_root_node()

        if search_budget is None and time_budget is not None:
            self.budget_type = 'time'
            self.budget = time_budget
        elif search_budget is not None and time_budget is None:
            self.budget_type = 'search'
            self.budget = search_budget
        else:
            raise ValueError('Either time_budget xor search_budget should be None, not neither or both')

        self.history = []
        super().__init__(name)

    @staticmethod
    def load_model(model_path, batch_size):
        model = load_model(model_path)
        # first prediction takes more time
        model.predict(np.array([State.empty().to_numpy()]))
        return BatchEvaluator(model, batch_size)

    def set_root_node(self, state: State = None):
        if state is None:
            state = State.empty()

        if self.root is not None:
            self.root = self.root.find_state(state)

        if self.root is None:
            self.root = AlphaConnectNode(state, action_prob=1.0)

        if self.is_self_play:
            self.root.add_dirichlet_noise = True

            if self.root.is_played:
                self.root.add_dirichlet_noise_to_action_probs()

        self.root.parent = None

    def clear_session(self):
        K.clear_session()

    def decide(self, state: State):
        t0 = time.time()
        self.set_root_node(state)

        if self.budget_type == 'time':
            while time.time() - t0 < self.budget / 1000:
                self.root.search(self.model, self.exploration)
        else:
            for _ in range(self.budget):
                self.root.search(self.model, self.exploration)

        self.save_policy()
        action = self.root.sample_action(self.temperature(state))
        return action

    def temperature(self, state: State):
        """AlphaGo lowers the temperature to infinitesimal after 30 moves

        Connect Four is a smaller game, so we use 16 moves

        When playing for real an infinitesimal temperature is always used
        """
        if self.is_self_play and state.number_of_stones < 16:
            return self._temperature
        return None

    def save_policy(self):
        self.history.append({
            'policy': self.root.policy(1.0),
            'total_value': self.root.total_value,
            'visit_count': self.root.visit_count
        })
