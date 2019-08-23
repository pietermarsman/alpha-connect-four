import math
import random
from typing import Dict, Union, List

import numpy as np

from analyzer import player_value
from state import Action, State, Color


class MiniMaxNode(object):
    def __init__(self, state, player_color, state_color=None, parent=None):
        self.state = state  # type: State
        self.player_color = player_color  # type: Color
        if state_color is None:
            self.state_color = player_color
        else:
            self.state_color = state_color
        self.parent = parent
        self.children = None
        self.value = player_value(self.state, self.player_color)

    def expand(self):
        if not self.state.is_end_of_game():
            self.children = {
                action: MiniMaxNode(self.state.take_action(action), self.player_color, self.state_color.other(), self)
                for action in self.state.allowed_actions}
        else:
            self.children = {}
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


class MonteCarloNode(object):
    def __init__(self, state: State, parent=None, exploration=1.0):
        self.state = state
        self.parent = parent  # type: Union['MonteCarloNode', None]
        self.exploration = exploration
        self.children = {}  # type: Dict[Action, MonteCarloNode]
        self.is_played = False
        self.visit_count = 0
        self.white_wins = 0
        self.brown_wins = 0

    def __str__(self):
        return 'Node(w=%d, b=%d, n=%d, uct=%.2f)' % (self.white_wins, self.brown_wins, self.visit_count, self.uct())

    def best_action(self):
        action, _ = max(self.children.items(), key=lambda x: x[1].visit_count)
        return action

    def search(self):
        selected_node = self.select()
        expanded_node = selected_node.expand()
        final_state = expanded_node.simulate()
        expanded_node.propagate(final_state)

    def select(self) -> 'MonteCarloNode':
        if self.is_played and not self.state.is_end_of_game():
            child = max(self.children.values(), key=lambda c: c.uct())
            return child.select()
        else:
            return self

    def uct(self):
        """Upper confidence bound for trees"""
        visit_count = self.visit_count + 0.001
        value = self._state_player_reward() / visit_count
        exploration = math.sqrt((2.0 * math.log(self.parent.visit_count)) / visit_count)
        return value + self.exploration * exploration

    def _state_player_reward(self):
        if self.state.next_color == Color.WHITE:
            return self.brown_wins
        else:
            return self.white_wins

    def expand(self) -> 'MonteCarloNode':
        self.is_played = True
        if not self.state.is_end_of_game():
            for action in self.state.allowed_actions:
                state = self.state.take_action(action)
                self.children[action] = MonteCarloNode(state, self, self.exploration)
            return random.choice(self.unvisited_children())
        else:
            return self

    def unvisited_children(self) -> 'List[MonteCarloNode]':
        return [child for child in self.children.values() if child.visit_count == 0]

    def simulate(self) -> 'State':
        state = self.state
        while not state.is_end_of_game():
            action = random.choice(list(state.allowed_actions))
            state = state.take_action(action)
        return state

    def propagate(self, final_state: State):
        self.visit_count += 1
        self.white_wins += final_state.winner == Color.WHITE
        self.brown_wins += final_state.winner == Color.BROWN
        if self.parent is not None:
            self.parent.propagate(final_state)

    def find_state(self, state: State):
        if self.state.number_of_stones < state.number_of_stones:
            for child in self.children.values():
                new_state = child.find_state(state)
                if new_state is not None:
                    return new_state
        elif self.state.number_of_stones == state.number_of_stones:
            if self.state == state:
                return self

        return None


class AlphaConnectNode(object):
    def __init__(self, state: State, parent=None, action_prob=None):
        self.state = state
        self.parent = parent  # type: Union[AlphaConnectNode, None]
        self.children = {}  # type: Dict[Action, AlphaConnectNode]
        self.is_played = False

        self.visit_count = 1
        self.total_value = 0
        self.action_prob = action_prob

    def __str__(self):
        return 'Node(prior=%.2f, value=%.2f/%d=%.2f)' % \
               (self.action_prob, self.total_value, self.visit_count, self.total_value / self.visit_count)

    def search(self, model: 'BatchEvaluator', c_puct: float):
        """Do a single MCTS search, with a select, expand, simulate and backup phase

        :param c_puct: exploration constant, higher is more exploration
        """
        selected_node = self.select(c_puct)
        selected_node.expand()
        self.lazy_evaluate_and_backup(model)

    def select(self, c_puct: float) -> 'AlphaConnectNode':
        if self.is_played and not self.state.is_end_of_game():
            child = max(self.children.values(), key=lambda child: child.puct(c_puct))
            return child.select(c_puct)
        else:
            return self

    def puct(self, c_puct: float):
        if self.parent is None:
            return 0.0

        average_value = self.total_value / self.visit_count
        exploration = self.action_prob * (math.sqrt(self.parent.visit_count) / self.visit_count)
        return average_value + c_puct * exploration

    def expand(self):
        if not self.state.is_end_of_game():
            for action in self.state.allowed_actions:
                state = self.state.take_action(action)
                self.children[action] = AlphaConnectNode(state, self, action_prob=0.0)
        self.is_played = True

    def lazy_evaluate_and_backup(self, model: 'BatchEvaluator'):
        model.simulate(self, callback=self.backup)

    def backup(self, value: float, action_probs: Union[None, Dict[Action, float]]):
        if not self.state.is_end_of_game() and action_probs is not None:
            for action in self.state.allowed_actions:
                self.children[action].action_prob = action_probs[action]
            if self.state.number_of_stones == 0:
                self._add_dirichlet_noise_to_action_prob()

        self.visit_count += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.backup(-value, None)

    def _add_dirichlet_noise_to_action_prob(self):
        """Additional dirichlet noise is added to empty state for additional exploration

        Dir(0.03) is highly skewed, putting almost all random weight onto a single action.
        """
        dirichlet_noise = np.random.dirichlet([0.03 for _ in range(len(self.children))])
        for action, noise in zip(self.children, dirichlet_noise.tolist()):
            self.children[action].action_prob = self.children[action].action_prob * .75 + noise * .25

    def find_state(self, state: State):
        if self.state.number_of_stones < state.number_of_stones:
            for child in self.children.values():
                new_state = child.find_state(state)
                if new_state is not None:
                    return new_state
        elif self.state.number_of_stones == state.number_of_stones:
            if self.state == state:
                return self

        return None

    def sample_action(self, temperature):
        actions, probabilities = zip(*self.policy(temperature).items())
        action = random.choices(list(actions), list(probabilities))[0]
        return action

    def policy(self, temperature) -> Dict[Action, float]:
        raw_policy = {action: node.exponentiated_visit_count(temperature) for action, node in self.children.items()}
        sum_policy = sum(raw_policy.values())
        return {action: policy_value / sum_policy for action, policy_value in raw_policy.items()}

    def exponentiated_visit_count(self, temperature) -> float:
        return self.visit_count ** (1.0 / temperature)


class BatchEvaluator(object):
    """Evaluate multiple states in batches"""

    def __init__(self, model, batch_size=8):
        self.model = model
        self.batch_size = batch_size
        self.queue = []

    def simulate(self, node: 'AlphaConnectNode', callback):
        if node.state.is_end_of_game():
            state_value = self.evaluate_final_state(node)
            callback(state_value, None)
        else:
            self.queue.append((node, callback))

        if len(self.queue) >= self.batch_size:
            nodes, callbacks = zip(*self.queue)
            array = np.concatenate(list(map(lambda node: node.state.to_numpy(batch=True), nodes)))
            pred_actions, pred_value = self.model.predict(array)

            for i, callback in enumerate(callbacks):
                state_value = pred_value[i].item()
                action_probs = dict(zip(Action.iter_actions(), pred_actions[i]))
                callback(state_value, action_probs)

            self.queue = []

    @staticmethod
    def evaluate_final_state(node):
        if node.state.winner == node.state.next_color:
            # game is already done, previous player is winner so current player gets a reward of -1
            state_value = -1.0
        elif node.state.winner == node.state.next_color.other():
            state_value = 1.0
        else:
            state_value = 0.0
        return state_value
