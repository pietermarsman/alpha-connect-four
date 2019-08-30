import os
from shutil import rmtree

import pytest

from util import list_files


@pytest.fixture
def directory_hierarchy():
    directory = 'test-dir'
    os.makedirs(directory)
    for i in range(100):
        subdirectory = '%.6d' % i
        os.makedirs(os.path.join(directory, subdirectory))
        open(os.path.join(directory, subdirectory, 'test.txt'), 'w').close()
    yield directory

    rmtree(directory)


def test_list_files_in_order(directory_hierarchy):
    files = list(list_files(directory_hierarchy))
    assert list(sorted(files)) == files
