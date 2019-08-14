import math
import random
from typing import Dict, Union, List

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
