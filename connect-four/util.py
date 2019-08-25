import os
from typing import Iterable

from state import Color, State


def list_files(data_dir, extension=None) -> Iterable[str]:
    for directory_path, _, file_names in os.walk(data_dir):
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
