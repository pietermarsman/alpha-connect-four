from itertools import product

from state import State, Color, FOUR, _lines_on_one_axis, _lines_on_one_diagonal, \
    _lines_on_two_diagonals, Action


def test_all_actions_are_possible_in_empty_state():
    state = State.empty()
    actions = state.allowed_actions
    assert set(product(range(FOUR), range(FOUR))) == actions


def test_action_in_empty_state_has_single_stone():
    state = State.empty()
    action = Action(0, 0)
    new_state = state.take_action(action)
    assert 1 == sum([stone is Color.WHITE for stone in new_state.stones.values()])


def test_action_changes_next_player():
    state = State.empty()
    state = state.take_action(Action(3, 3))
    state = state.take_action(Action(3, 3))
    assert Color.WHITE is state.next_color


def test_48_solutions_one_one_axis():
    one_axis_solutions = _lines_on_one_axis()
    assert 48 == len(one_axis_solutions)


def test_24_solutions_on_one_diagonal():
    one_diagonal_solutions = _lines_on_one_diagonal()
    assert 24 == len(one_diagonal_solutions)


def test_2_solutions_on_two_diagonals():
    two_diagonal_solutions = _lines_on_two_diagonals()
    assert 4 == len(two_diagonal_solutions)


def test_empty_board_has_no_winner():
    state = State.empty()
    assert not state.has_winner()


def test_simple_state_has_winner():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(1, 0))
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(1, 0))
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(1, 0))
    state = state.take_action(Action(0, 0))
    assert state.has_winner
