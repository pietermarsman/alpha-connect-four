from operator import sub

from connectfour3d import State, Stone


def count_lines(state: State, player_stone: Stone):
    value = [0, 0, 0, 0, 0]
    for connected_stones in state.connected_stones_owned_by(player_stone):
        n_player_stones = sum([stone is player_stone for stone in connected_stones])
        value[4 - n_player_stones] += 1
    return tuple(value)


def player_value(state: State, player_stone: Stone):
    my_value = count_lines(state, player_stone)
    other_value = count_lines(state, player_stone.other())
    return tuple(map(sub, my_value, other_value))
