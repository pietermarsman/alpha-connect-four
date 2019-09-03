import random
from itertools import product

import numpy as np
import pytest

from state import State, Color, FOUR, _lines_on_one_axis, _lines_on_one_diagonal, \
    _lines_on_two_diagonals, Action, _lines, Augmentation, Rotation, Position


@pytest.fixture
def random_state():
    state = State.empty()
    for _ in range(6):
        action = random.choice(list(state.allowed_actions))
        state = state.take_action(action)

    return state


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


def test_lines_have_no_duplicates():
    lines = _lines().values()
    assert list(sorted(set(lines))) == list(sorted(lines))


def test_lines_have_no_inverse_duplicates():
    lines = list(sorted(_lines().values()))
    inverse_lines = [tuple((line[i] for i in range(FOUR - 1, -1, -1))) for line in lines]
    assert set() == set(lines) & set(inverse_lines)


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


def test_winner_on_diagonal_line_along_side():
    state = State.empty()
    state = state.take_action(Action(0, 3))  # white
    state = state.take_action(Action(0, 2))  # brown
    state = state.take_action(Action(0, 2))  # white
    state = state.take_action(Action(0, 1))  # brown
    state = state.take_action(Action(0, 0))  # white
    state = state.take_action(Action(0, 1))  # brown
    state = state.take_action(Action(0, 1))  # white
    state = state.take_action(Action(0, 0))  # brown
    state = state.take_action(Action(1, 0))  # white
    state = state.take_action(Action(0, 0))  # brown
    state = state.take_action(Action(0, 0))  # white
    assert state.is_end_of_game()
    assert state.winner is Color.WHITE


def test_action_to_int():
    action = Action(2, 3)
    assert action == Action.from_int(action.to_int())


def test_rotating_four_quarters_is_same():
    old_action = Action(1, 3)
    action = old_action
    for _ in range(4):
        action = action.augment(Augmentation(Rotation.QUARTER, False))

    assert old_action == action


def test_flipping_twice_is_same():
    old_action = Action(0, 0)
    action = old_action
    action = action.augment(Augmentation(Rotation.NO, True))
    action = action.augment(Augmentation(Rotation.NO, True))
    assert old_action == action


def test_state_to_numpy_without_augmentation():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    state = state.take_action(Action(0, 1))

    arr = state.to_numpy()
    expected_white = np.zeros((4, 4)).astype(bool)
    expected_white[0, 0] = True
    expected_black = np.zeros((4, 4)).astype(bool)
    expected_black[0, 1] = True

    assert expected_white.tolist() == arr[:, :, 0, 0].tolist()
    assert expected_black.tolist() == arr[:, :, 0, 1].tolist()


def test_state_to_numpy_with_quarter_rotation():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    arr = state.to_numpy(Augmentation(Rotation.QUARTER, False))

    expected = np.zeros((4, 4)).astype(bool)
    expected[3, 0] = True
    assert expected.tolist() == arr[:, :, 0, 1].tolist()


def test_state_to_numpy_with_three_quarter_rotation_and_x_flip():
    state = State.empty()
    state = state.take_action(Action(0, 0))
    arr = state.to_numpy(Augmentation(Rotation.THREE_QUARTER, True))

    expected = np.zeros((4, 4)).astype(bool)
    expected[3, 3] = True
    assert expected.tolist() == arr[:, :, 0, 1].tolist()


def test_position_rotation():
    position = Position(0, 3, 4).augment(Augmentation(Rotation.HALF, False))
    assert Position(3, 0, 4) == position


def test_each_augmentation_result_in_unique_states(random_state):
    augmented_states = []
    for augmentation in Augmentation.iter_augmentations():
        augmented_states.append(random_state.to_numpy(augmentation).tostring())

    assert len(augmented_states) == len(set(augmented_states))


def test_white_lines_is_updated_after_action():
    state = State.empty()
    white_action = Action(2, 3)
    white_position = Position.from_action_and_height(white_action, 0)
    brown_action = Action(1, 2)
    brown_position = Position.from_action_and_height(brown_action, 0)
    intersection_postition = Position(white_position.x, brown_position.y, 0)

    state = state.take_action(white_action)
    state = state.take_action(brown_action)

    # white position
    for line_i, _ in State.POSITION_TO_LINES[white_position]:
        assert 1 == state.white_lines[line_i]
        assert 0 == state.brown_lines[line_i]

    assert 4 == state.white_lines_free[white_position]
    assert 0 == state.brown_lines_free[white_position]

    assert 1 == state.white_max_line[white_position]
    assert 0 == state.brown_max_line[white_position]

    # brown position
    for line_i, _ in State.POSITION_TO_LINES[brown_position]:
        assert 1 == state.brown_lines[line_i]
        assert 0 == state.white_lines[line_i]

    assert 4 == state.brown_lines_free[brown_position]
    assert 0 == state.white_lines_free[brown_position]

    assert 1 == state.brown_max_line[brown_position]
    assert 0 == state.white_max_line[brown_position]

    # intersection
    assert 3 == state.brown_lines_free[intersection_postition]
    assert 3 == state.white_lines_free[intersection_postition]

    assert 1 == state.brown_max_line[intersection_postition]
    assert 1 == state.white_max_line[intersection_postition]


def test_regression_str_with_top_plane_stones():
    actions_history = [Action.from_hex(i) for i in '0cf35aa55ae9699663cb8c7447f8ec']
    state = State.empty().take_actions(actions_history)
    str(state)
