import math
import random
from typing import Dict, Union, List, Tuple

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

    def unvisited_children(self) -> 'List[MonteCarloNode]':
        return [child for child in self.children.values() if child.visit_count == 0]

    def expand(self) -> 'MonteCarloNode':
        self.is_played = True
        if not self.state.is_end_of_game():
            for action in self.state.allowed_actions:
                state = self.state.take_action(action)
                self.children[action] = MonteCarloNode(state, self, self.exploration)
            return random.choice(self.unvisited_children())
        else:
            return self

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
    def __init__(self, state: State, model, parent=None, action_prob=None, c_puct=1.0, temperature=1.0):
        self.state = state
        self.model = model
        self.parent = parent  # type: Union[AlphaConnectNode, None]
        self.c_puct = c_puct
        self.temperature = temperature
        self.children = {}  # type: Dict[Action, AlphaConnectNode]
        self.is_played = False

        self.visit_count = 1
        self.total_value = 0
        self.action_prob = action_prob

    def __str__(self):
        return 'Node(%d / %d = %.2f)' % (self.total_value, self.visit_count, self.total_value / self.visit_count)

    def search(self):
        selected_node = self.select()
        value = selected_node.expand_and_simulate()
        selected_node.propagate(value)

    def select(self) -> 'AlphaConnectNode':
        if self.is_played and not self.state.is_end_of_game():
            child = max(self.children.values(), key=lambda c: c.puct())
            return child.select()
        else:
            return self

    def puct(self):
        average_value = self.total_value / self.visit_count
        exploration = self.action_prob * (math.sqrt(self.parent.visit_count) / self.visit_count)
        return average_value + self.c_puct * exploration

    def expand_and_simulate(self):
        self.is_played = True
        state_value, action_probs = self.simulate()
        if not self.state.is_end_of_game():
            for action in self.state.allowed_actions:
                action_prob = action_probs[action]
                state = self.state.take_action(action)
                self.children[action] = AlphaConnectNode(state, self.model, self, action_prob=action_prob,
                                                         c_puct=self.c_puct, temperature=self.temperature)
        return state_value

    def simulate(self) -> Tuple[float, Dict[Action, float]]:
        """Get value for next color and action probabilities"""
        if self.state.is_end_of_game():
            if self.state.winner == self.state.next_color:
                # game is already done, previous player is winner so current player gets a reward of -1
                state_value = -1.0
            elif self.state.winner == self.state.next_color.other():
                state_value = 1.0
            else:
                state_value = 0.0
            action_probs = None
        else:
            # todo predict for multiple states in batches
            array = self.state.to_numpy(batch=True)
            pred_actions, pred_value = self.model.predict(array)
            state_value = pred_value.item()
            action_probs = dict(zip(Action.iter_actions(), pred_actions[0]))

        return state_value, action_probs

    def propagate(self, value: float):
        self.visit_count += 1
        self.total_value += value
        if self.parent is not None:
            self.parent.propagate(-value)

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

    def sample_action(self):
        actions, probabilities = zip(*self.policy().items())
        action = random.choices(list(actions), list(probabilities))[0]
        return action

    def policy(self) -> Dict[Action, float]:
        raw_policy = {action: node.exponentiated_visit_count() for action, node in self.children.items()}
        sum_policy = sum(raw_policy.values())
        return {action: policy_value / sum_policy for action, policy_value in raw_policy.items()}

    def exponentiated_visit_count(self) -> float:
        return self.visit_count ** (1.0 / self.temperature)
