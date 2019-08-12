from operator import sub

from state import Color, State, FOUR


def count_lines(state: State):
    brown_value = [0, 0, 0, 0, 0]
    white_value = [0, 0, 0, 0, 0]
    for line_i, brown_count in state.brown_lines.items():
        white_count = state.white_lines[line_i]
        if brown_count == 0 and white_count > 0:
            white_value[FOUR - white_count] += 1
        elif white_count == 0 and brown_count > 0:
            brown_value[FOUR - brown_count] += 1
    return brown_value, white_value


def player_value(state: State, color: Color):
    brown_value, white_value = count_lines(state)
    if color is Color.BROWN:
        return tuple(map(sub, brown_value, white_value))
    elif color is Color.WHITE:
        return tuple(map(sub, white_value, brown_value))
