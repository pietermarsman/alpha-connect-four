import os
from typing import Iterable, Dict, Any

from state import Color, State, Action, FOUR


def list_files(data_dir, extension=None) -> Iterable[str]:
    for directory_path, subdirectories, file_names in os.walk(data_dir):
        subdirectories.sort()  # reading data files relies on having the input in alphabetical order
        for file_name in sorted(file_names):
            if extension is None or file_name.endswith(extension):
                yield os.path.abspath(os.path.join(directory_path, file_name))


def replace_extension(path: str, new_extension: str):
    name, ext = os.path.splitext(path)
    new_extension = new_extension.strip('.')
    return name + '.' + new_extension


def winner_value(winner: Color, state: State):
    if winner is state.next_color:
        return 1.0
    elif winner is state.next_color.other():
        return -1.0
    else:
        return 0.0


def format_in_action_grid(values: Dict[Action, Any], cell_format='{:4.0%}', default_value='    ', seperator=' '):
    grid = [[default_value for _ in range(FOUR)] for _ in range(FOUR)]
    for action, value in values.items():
        grid[action.y][action.x] = cell_format.format(value)

    return '\n'.join([seperator.join(row) for row in grid])
