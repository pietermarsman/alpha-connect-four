from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from typing import Tuple

FOUR = 4


def _position_solutions():
    solutions = _solutions()
    positions = {}
    for solution_i, solution in enumerate(solutions):
        for position_i, position in enumerate(solution):
            if position in positions:
                positions[position].append((solution_i, position_i))
            else:
                positions[position] = [(solution_i, position_i)]
            positions[position] = positions[position].get(position, []) + [position]
    return positions


def _solutions():
    return _solutions_on_one_axis() + _solutions_on_one_diagonal() + _solutions_on_two_diagonals()


def _solutions_on_one_axis():
    base_solutions = [[(pos1, pos2, pos3) for pos3 in range(FOUR)]
                      for pos1, pos2 in product(range(FOUR), range(FOUR))]
    all_solutions = [tuple(Position(pos[i], pos[j], pos[k]) for pos in solution)
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _solutions_on_one_diagonal():
    base_solutions = [[(pos1, pos2, pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)] + \
                     [[(pos1, FOUR-1-pos2, FOUR-1-pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)]
    all_solutions = [tuple(Position(pos[i], pos[j], pos[k]) for pos in solution)
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _solutions_on_two_diagonals():
    all_solutions = [tuple(Position(pos1, pos1, pos1) for pos1 in range(FOUR)),
                     tuple(Position(pos1, FOUR - 1 - pos1, FOUR - 1 - pos1) for pos1 in range(FOUR))]
    return all_solutions


class Stone(Enum):
    NONE = 0
    BROWN = 1
    WHITE = 2

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


Action = namedtuple('Action', ['x', 'y'])
_Position = namedtuple('Position', ['x', 'y', 'z'])


class Position(_Position):
    @classmethod
    def iter_positions(cls):
        for x, y, z in product(range(FOUR), range(FOUR), range(FOUR)):
            yield Position(x, y, z)


class State(object):
    """State of a 3d connect four game

    Uses 3-dimensional coordinate system: x, y, z
    """
    SOLUTIONS = _solutions()
    POSITIONS = {
        pos: [(solution_i, pos_i)]
        for solution_i, solution in enumerate(SOLUTIONS)
        for pos_i, pos in enumerate(solution)
    }

    def __init__(self, stones: dict = None, next_stone=Stone.WHITE, connected_stones=None, connected_owned_by=None):
        if stones is None:
            self._stones = {pos: Stone.NONE for pos in Position.iter_positions()}
        else:
            self._stones = stones

        if connected_stones is None:
            self._connected_stones = [[self._stones[pos] for pos in solution] for solution in self.SOLUTIONS]
        else:
            self._connected_stones = connected_stones

        if connected_owned_by is None:
            self._connected_stones_owned_by = {}
        else:
            self._connected_stones_owned_by = connected_owned_by

        self.next_stone = next_stone

    @classmethod
    def from_earlier_state(cls, previous_state: 'State', action: Action):
        pin = previous_state._vertical_layer(action.x, action.y)
        height = min((z for (_, _, z), stone in pin.items() if stone is Stone.NONE))
        assert height < FOUR, 'Invalid move'
        pos = Position(action.x, action.y, height)
        new_state = previous_state._stones.copy()
        new_state[(action.x, action.y, height)] = previous_state.next_stone

        connected_stones = previous_state._connected_stones
        for (solution_i, position_i) in cls.POSITIONS[pos]:
            connected_stones[solution_i][position_i] = previous_state.next_stone

        return State(new_state, previous_state.next_stone.other())

    @property
    def stones(self):
        return self._stones

    def __str__(self):
        state_array = [[['?' for _ in range(FOUR)] for _ in range(FOUR)] for _ in range(FOUR)]
        for (x, y, z), stone in self._stones.items():
            state_array[z][y][x] = str(stone)
        rows = [[''.join(row) for row in layer] for layer in reversed(state_array)]
        layers = ['\n'.join(layer_rows) for layer_rows in rows]
        board = '\n\n'.join(layers)
        return board

    def __getitem__(self, pos):
        return self._stones[pos]

    def take_action(self, pos: Tuple[int, int]):
        return State.from_earlier_state(self, Action(*pos))

    def _vertical_layer(self, x, y):
        assert 0 <= x < FOUR and 0 <= y < FOUR
        return {(sx, sy, sz): stone for (sx, sy, sz), stone in self._stones.items() if sx == x and sy == y}

    def _height(self, x, y):
        for z in range(FOUR):
            if self[(x, y, z)] == Stone.NONE:
                return z
        return FOUR

    def possible_actions(self):
        top_layer = self._horizontal_layer(3)
        actions = {(x, y) for (x, y, _), stone in top_layer.items() if stone == Stone.NONE}
        return actions

    def _horizontal_layer(self, height):
        assert 0 <= height < FOUR
        return {(x, y, z): stone for (x, y, z), stone in self._stones.items() if z == height}

    def is_end_of_game(self):
        return self.has_winner() or self._is_full()

    def has_winner(self):
        return self.winner() is not None

    def winner(self):
        for connected_stones in self._connected_stones:
            if all([stone == Stone.BROWN for stone in connected_stones]):
                return Stone.BROWN
            if all([stone == Stone.WHITE for stone in connected_stones]):
                return Stone.WHITE
        return None

    def connected_stones_owned_by(self, player_stone: Stone):
        if player_stone not in self._connected_stones_owned_by:
            player_connected_stones = []
            other_stone = player_stone.other()
            for connected_stones in self._connected_stones:
                not_any_other = all((stone != other_stone for stone in connected_stones))
                if not_any_other:
                    player_connected_stones.append(connected_stones)
            self._connected_stones_owned_by[player_stone] = player_connected_stones

        return self._connected_stones_owned_by[player_stone]

    def _is_full(self):
        return all(stone is not Stone.NONE for stone in self._stones.values())
