from itertools import product

from state import ConnectFour3D, Stone, FOUR, _solutions_on_one_axis, _solutions_on_one_diagonal, \
    _solutions_on_two_diagonals


def test_all_actions_are_possible_in_empty_state():
    state = ConnectFour3D()
    actions = state.possible_actions()
    assert set(product(range(FOUR), range(FOUR))) == actions


def test_bottom_layer_is_empty_in_empty_state():
    state = ConnectFour3D()
    expected = {(x, y, 0): Stone.NONE for x in range(FOUR) for y in range(FOUR)}
    assert expected == state._horizontal_layer(0)


def test_action_in_empty_state_has_single_stone():
    state = ConnectFour3D(Stone.WHITE)
    action = (0, 0)
    new_state = state.take_action(action)
    assert 1 == sum([stone is Stone.WHITE for stone in new_state.stones.values()])


def test_action_changes_next_player():
    state = ConnectFour3D(Stone.BROWN)
    new_state = state.take_action((3, 3))
    assert Stone.WHITE is new_state.next_stone


def test_48_solutions_one_one_axis():
    one_axis_solutions = _solutions_on_one_axis()
    assert 48 == len(one_axis_solutions)


def test_24_solutions_on_one_diagonal():
    one_diagonal_solutions = _solutions_on_one_diagonal()
    assert 24 == len(one_diagonal_solutions)


def test_2_solutions_on_two_diagonals():
    two_diagonal_solutions = _solutions_on_two_diagonals()
    assert 2 == len(two_diagonal_solutions)


def test_board_full_of_brown_stones_has_winner():
    state = ConnectFour3D()
    for pos in state.stones:
        state.stones[pos] = Stone.BROWN
    assert state._has_winner()


def test_empty_board_has_no_winner():
    state = ConnectFour3D()
    assert not state._has_winner()


def test_simple_state_has_winner():
    state = ConnectFour3D()
    state = state.take_action((0, 0))
    state = state.take_action((1, 0))
    state = state.take_action((0, 0))
    state = state.take_action((1, 0))
    state = state.take_action((0, 0))
    state = state.take_action((1, 0))
    state = state.take_action((0, 0))
    assert state._has_winner()
