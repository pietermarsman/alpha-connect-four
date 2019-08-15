from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from typing import Dict, Set, NamedTuple, Union

FOUR = 4


def _position_to_lines():
    lines = _lines()
    positions = {}
    for line_i, line in lines.items():
        for position_i, position in enumerate(line):
            if position in positions:
                positions[position].append((line_i, position_i))
            else:
                positions[position] = [(line_i, position_i)]
    return positions


def _lines():
    return dict(enumerate(_lines_on_one_axis() + _lines_on_one_diagonal() + _lines_on_two_diagonals()))


def _lines_on_one_axis():
    base_solutions = [[(pos1, pos2, pos3) for pos3 in range(FOUR)]
                      for pos1, pos2 in product(range(FOUR), range(FOUR))]
    all_solutions = [tuple(sorted(Position(pos[i], pos[j], pos[k]) for pos in solution))
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _lines_on_one_diagonal():
    base_solutions = [[(pos1, pos2, pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)] + \
                     [[(pos1, pos2, FOUR - 1 - pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)]
    all_solutions = [tuple(sorted(Position(pos[i], pos[j], pos[k]) for pos in solution))
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _lines_on_two_diagonals():
    all_solutions = [
        tuple(sorted(Position(pos1, pos1, pos1) for pos1 in range(FOUR))),
        tuple(sorted(Position(pos1, FOUR - 1 - pos1, pos1) for pos1 in range(FOUR))),
        tuple(sorted(Position(FOUR - 1 - pos1, pos1, pos1) for pos1 in range(FOUR))),
        tuple(sorted(Position(FOUR - 1 - pos1, FOUR - 1 - pos1, pos1) for pos1 in range(FOUR)))
    ]
    return all_solutions


class Color(Enum):
    NONE = 0
    BROWN = 1
    WHITE = 2

    @classmethod
    def iter_colors(cls):
        yield cls.BROWN
        yield cls.WHITE

    def other(self):
        assert self is not self.NONE
        if self is self.BROWN:
            return self.WHITE
        else:
            return self.BROWN

    def __str__(self):
        if self is self.NONE:
            return '.'
        elif self is self.BROWN:
            return 'b'
        else:
            return 'w'


_Action = namedtuple('Action', ['x', 'y'])


class Action(_Action):
    @classmethod
    def iter_actions(cls):
        for x, y in product(range(FOUR), range(FOUR)):
            yield Action(x, y)

    @classmethod
    def from_int(cls, i):
        x, y = divmod(i, 4)
        return cls(x, y)

    def to_int(self):
        return self.x * 4 + self.y


_Position = namedtuple('Position', ['x', 'y', 'z'])


class Position(_Position):
    @classmethod
    def iter_positions(cls):
        for x, y, z in product(range(FOUR), range(FOUR), range(FOUR)):
            yield Position(x, y, z)

    @classmethod
    def from_action_and_height(cls, action: Action, height: int):
        return Position(action.x, action.y, height)

    def to_action(self):
        return Action(self.x, self.y)


_State = NamedTuple('State', [
    ('stones', Dict[Position, Color]),
    ('next_color', Color),
    ('pin_height', Dict[Action, int]),
    ('allowed_actions', Set[Action]),
    ('number_of_stones', int),
    ('brown_lines', Dict[int, int]),
    ('white_lines', Dict[int, int]),
    ('winner', Union[Color, None])
])


class State(_State):
    """State of a 3d connect four game

    Uses 3-dimensional coordinate system: x, y, z
    """
    LINES = _lines()
    POSITION_TO_LINES = _position_to_lines()

    @classmethod
    def empty(cls) -> 'State':
        stones = {pos: Color.NONE for pos in Position.iter_positions()}
        next_color = Color.WHITE
        pin_height = {action: 0 for action in Action.iter_actions()}
        allowed_actions = set(pin_height.keys())
        number_of_stones = 0
        winner = None
        brown_lines = {line_i: 0 for line_i in cls.LINES.keys()}
        white_lines = {line_i: 0 for line_i in cls.LINES.keys()}
        return cls(stones, next_color, pin_height, allowed_actions, number_of_stones, brown_lines, white_lines,
                   winner)

    def take_action(self, action: Action) -> 'State':
        assert action in self.allowed_actions
        assert not self.has_winner()

        stones = self.stones.copy()  # does not deepcopy keys, but these never have to change
        next_color = self.next_color
        pin_height = self.pin_height.copy()  # does not deepcopy keys, but these never have to change
        allowed_actions = self.allowed_actions.copy()  # does not deepcopy items, but these never have to change
        number_of_stones = self.number_of_stones
        brown_lines = self.brown_lines.copy()
        white_lines = self.white_lines.copy()
        winner = self.winner

        position = Position.from_action_and_height(action, self.pin_height[action])
        stones[position] = self.next_color
        next_color = next_color.other()
        pin_height[action] += 1
        if pin_height[action] == FOUR:
            allowed_actions.remove(action)
        number_of_stones += 1
        affected_lines = self.POSITION_TO_LINES[position]
        for line_i, line_ii in affected_lines:
            if self.next_color == Color.BROWN:
                brown_lines[line_i] += 1
                if brown_lines[line_i] == FOUR:
                    winner = Color.BROWN
            else:
                white_lines[line_i] += 1
                if white_lines[line_i] == FOUR:
                    winner = Color.WHITE

        return State(stones, next_color, pin_height, allowed_actions, number_of_stones, brown_lines,
                     white_lines, winner)

    def __str__(self):
        state_array = [[['?' for _ in range(FOUR)] for _ in range(FOUR)] for _ in range(FOUR)]
        for (x, y, z), stone in self.stones.items():
            state_array[z][y][x] = str(stone)
        rows = [[''.join(row) for row in layer] for layer in reversed(state_array)]
        layers = ['\n'.join(layer_rows) for layer_rows in rows]
        board = '\n\n'.join(layers)
        return board

    def is_end_of_game(self):
        is_full = all(stone is not Color.NONE for stone in self.stones.values())
        return self.winner is not None or is_full

    def has_winner(self):
        return self.winner is not None
