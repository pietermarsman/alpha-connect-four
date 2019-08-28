from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from typing import Dict, Set, NamedTuple, Union

import numpy as np

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


class Rotation(Enum):
    NO = 0
    QUARTER = 1
    HALF = 2
    THREE_QUARTER = 3

    @classmethod
    def iter_rotations(cls):
        yield from [Rotation.NO, Rotation.QUARTER, Rotation.HALF, Rotation.THREE_QUARTER]


_Augmentation = NamedTuple('Augmentation', [
    ('rotation', Rotation),
    ('flip_x', bool),
])


class Augmentation(_Augmentation):
    @classmethod
    def identity(cls):
        return Augmentation(Rotation.NO, False)

    @classmethod
    def iter_augmentations(cls):
        for rotation in Rotation.iter_rotations():
            for flip_x in [False, True]:
                yield Augmentation(rotation, flip_x)


_Action = namedtuple('Action', ['x', 'y'])


class Action(_Action):
    @classmethod
    def iter_actions(cls):
        for x, y in product(range(FOUR), range(FOUR)):
            yield Action(x, y)

    @classmethod
    def from_int(cls, i):
        x, y = divmod(i, FOUR)
        return cls(x, y)

    def to_int(self):
        return self.x * FOUR + self.y

    @classmethod
    def from_hex(cls, i):
        x, y = divmod(int(i, base=FOUR * FOUR), FOUR)
        return cls(x, y)

    def to_hex(self):
        return hex(self.x * FOUR + self.y)[2:]

    def __str__(self):
        return self.to_hex()

    def augment(self, augmentation: Augmentation) -> 'Action':
        x, y = self.x, self.y

        for _ in range(augmentation.rotation.value):
            temp_y = y
            y = x
            x = FOUR - 1 - temp_y

        if augmentation.flip_x:
            x = FOUR - 1 - x

        return Action(x, y)


_Position = namedtuple('Position', ['x', 'y', 'z'])


class Position(_Position):
    @classmethod
    def iter_positions(cls):
        for x, y, z in product(range(FOUR), range(FOUR), range(FOUR)):
            yield Position(x, y, z)

    def augment(self, augmentation: Augmentation) -> 'Position':
        action = self.to_action().augment(augmentation)
        return Position(action.x, action.y, self.z)

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
    ('brown_lines_free', Dict[Position, int]),
    ('white_lines_free', Dict[Position, int]),
    ('brown_max_line', Dict[Position, int]),
    ('white_max_line', Dict[Position, int]),
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
        brown_lines_free = {pos: len(cls.POSITION_TO_LINES[pos]) for pos in Position.iter_positions()}
        white_lines_free = {pos: len(cls.POSITION_TO_LINES[pos]) for pos in Position.iter_positions()}
        brown_max_line = {pos: 0 for pos in Position.iter_positions()}
        white_max_line = {pos: 0 for pos in Position.iter_positions()}
        return cls(stones, next_color, pin_height, allowed_actions, number_of_stones, brown_lines, white_lines,
                   brown_lines_free, white_lines_free, brown_max_line, white_max_line, winner)

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
        brown_lines_free = self.brown_lines_free.copy()
        white_lines_free = self.white_lines_free.copy()
        brown_max_line = self.brown_max_line.copy()
        white_max_line = self.white_max_line.copy()
        winner = self.winner

        position = Position.from_action_and_height(action, self.pin_height[action])
        stones[position] = self.next_color
        next_color = next_color.other()
        pin_height[action] += 1
        if pin_height[action] == FOUR:
            allowed_actions.remove(action)
        number_of_stones += 1
        affected_lines = self.POSITION_TO_LINES[position]
        for line_i, _ in affected_lines:
            if self.next_color == Color.BROWN:
                brown_lines[line_i] += 1
                for line_pos in self.LINES[line_i]:
                    brown_max_line[line_pos] = max(brown_lines[line_i], brown_max_line[line_pos])
                    if brown_lines[line_i] == 1:
                        white_lines_free[line_pos] -= 1
                if brown_lines[line_i] == FOUR:
                    winner = Color.BROWN
            else:
                white_lines[line_i] += 1
                for line_pos in self.LINES[line_i]:
                    white_max_line[line_pos] = max(white_lines[line_i], white_max_line[line_pos])
                    if white_lines[line_i] == 1:
                        brown_lines_free[line_pos] -= 1
                if white_lines[line_i] == FOUR:
                    winner = Color.WHITE

        return State(stones, next_color, pin_height, allowed_actions, number_of_stones, brown_lines,
                     white_lines, brown_lines_free, white_lines_free, brown_max_line, white_max_line, winner)

    def __str__(self):
        state_array = [[['?' for _ in range(FOUR)] for _ in range(FOUR)] for _ in range(FOUR)]
        for (x, y, z), stone in self.stones.items():
            state_array[z][y][x] = str(stone)
        rows = [[' '.join(row) for row in layer] for layer in reversed(state_array)]
        layers = ['\n'.join(layer_rows) for layer_rows in rows]
        board = '\n\n'.join(layers)
        return board

    def is_end_of_game(self):
        is_full = all(stone is not Color.NONE for stone in self.stones.values())
        return self.winner is not None or is_full

    def has_winner(self):
        return self.winner is not None

    def to_numpy(self, augmentation: Augmentation = None, batch=False):
        if augmentation is None:
            arr = [[[self._encode_position(Position(x, y, z))
                     for z in range(FOUR)]
                    for y in range(FOUR)]
                   for x in range(FOUR)]
        else:
            mapping = {position.augment(augmentation): position for position in Position.iter_positions()}
            arr = [[[self._encode_position(mapping[Position(x, y, z)])
                     for z in range(FOUR)]
                    for y in range(FOUR)]
                   for x in range(FOUR)]

        if batch:
            arr = np.array([arr, ])
        return np.array(arr)

    def _encode_position(self, pos: Position):
        x, y, z = pos

        stone = self.stones[pos]
        reachable = z == self.pin_height[Action(x, y)]

        corner = (x == 0 or x == 3) and (y == 0 or y == 3)
        side = (x == 0 or x == 3 or y == 0 or y == 3) and not corner
        middle = not (corner or side)
        bottom = (z == 0)
        top = (z == 3)
        middle_z = not (bottom or top)

        if self.next_color == Color.WHITE:
            my_lines_free = self.white_lines_free[pos]
            other_lines_block = self.brown_lines_free[pos]
            my_max_line = self.white_max_line[pos]
            other_max_line = self.brown_max_line[pos]
        else:
            my_lines_free = self.brown_lines_free[pos]
            other_lines_block = self.white_lines_free[pos]
            my_max_line = self.brown_max_line[pos]
            other_max_line = self.white_max_line[pos]

        return (
            stone == self.next_color,
            stone == self.next_color.other(),
            reachable,
            corner,
            side,
            middle,
            bottom,
            top,
            middle_z,
            my_lines_free,
            other_lines_block,
            my_max_line,
            other_max_line
        )
