from enum import Enum
from itertools import product, permutations
from typing import Tuple

FOUR = 4


def _solutions():
    return _solutions_on_one_axis() + _solutions_on_one_diagonal() + _solutions_on_two_diagonals()


def _solutions_on_one_axis():
    base_solutions = [[(pos1, pos2, pos3) for pos3 in range(FOUR)]
                      for pos1, pos2 in product(range(FOUR), range(FOUR))]
    all_solutions = [tuple((pos[i], pos[j], pos[k]) for pos in solution)
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _solutions_on_one_diagonal():
    base_solutions = [[(pos1, pos2, pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)] + \
                     [[(pos1, FOUR-1-pos2, FOUR-1-pos2) for pos2 in range(FOUR)] for pos1 in range(FOUR)]
    all_solutions = [tuple((pos[i], pos[j], pos[k]) for pos in solution)
                     for solution in base_solutions for i, j, k in permutations(range(3))]
    return list(set(all_solutions))


def _solutions_on_two_diagonals():
    all_solutions = [tuple((pos1, pos1, pos1) for pos1 in range(FOUR)),
                     tuple((pos1, FOUR-1-pos1, FOUR-1-pos1) for pos1 in range(FOUR))]
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


class ConnectFour3D(object):
    """State of a 3d connect four game

    Uses 3-dimensional coordinate system: x, y, z
    """
    SOLUTIONS = _solutions()

    def __init__(self, next_stone=Stone.WHITE, stones: dict=None):
        if stones is None:
            self.stones = {pos: Stone.NONE for pos in product(range(FOUR), range(FOUR), range(FOUR))}
        else:
            self.stones = stones

        self.next_stone = next_stone

    def __str__(self):
        state_array = [[['?' for _ in range(FOUR)] for _ in range(FOUR)] for _ in range(FOUR)]
        for (x, y, z), stone in self.stones.items():
            state_array[z][y][x] = str(stone)
        rows = [[''.join(row) for row in layer] for layer in reversed(state_array)]
        layers = ['\n'.join(layer_rows) for layer_rows in rows]
        board = '\n\n'.join(layers)
        return board

    def __getitem__(self, pos):
        return self.stones[pos]

    def take_action(self, pos: Tuple[int, int]):
        x, y = pos
        pin = self._vertical_layer(x, y)
        height = min((z for (_, _, z), stone in pin.items() if stone is Stone.NONE))
        new_state = self.stones.copy()
        new_state[(x, y, height)] = self.next_stone
        return ConnectFour3D(self.next_stone.other(), new_state)

    def _vertical_layer(self, x, y):
        assert 0 <= x < FOUR and 0 <= y < FOUR
        return {(sx, sy, sz): stone for (sx, sy, sz), stone in self.stones.items() if sx == x and sy == y}

    def possible_actions(self):
        top_layer = self._horizontal_layer(3)
        actions = {(x, y) for (x, y, _), stone in top_layer.items() if stone == Stone.NONE}
        return actions

    def _horizontal_layer(self, height):
        assert 0 <= height < FOUR
        return {(x, y, z): stone for (x, y, z), stone in self.stones.items() if z == height}

    def is_end_of_game(self):
        return self._has_winner() or self._is_full()

    def _has_winner(self):
        return self.winner() is not None

    def winner(self):
        for connected_stones in self.connected_stones():
            if all([stone == Stone.BROWN for stone in connected_stones]):
                return Stone.BROWN
            if all([stone == Stone.WHITE for stone in connected_stones]):
                return Stone.WHITE
        return None

    def connected_stones(self):
        return [[self.stones[pos] for pos in solution] for solution in self.SOLUTIONS]

    def connected_stones_owned_by(self, player_stone: Stone):
        player_connected_stones = []
        other_stone = player_stone.other()
        for connected_stones in self.connected_stones():
            not_any_other = all((stone != other_stone for stone in connected_stones))
            if not_any_other:
                player_connected_stones.append(connected_stones)
        return player_connected_stones

    def _is_full(self):
        return all(stone is not Stone.NONE for stone in self.stones.values())
