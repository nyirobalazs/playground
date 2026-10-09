import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synthetic as S  # noqa: E402


@pytest.fixture(scope="session")
def rec():
    return S.make_recording(seed=1)


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_files")
    return S.write_files(str(d), seed=2)
