"""Global test config."""

import pytest
from rex import init_logger


@pytest.hookimpl
def pytest_configure(config):  # noqa: D103 ARG001
    init_logger('nsrdb', log_level='DEBUG')
