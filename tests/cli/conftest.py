"""runner fixture for cli tests."""

import pytest
from click.testing import CliRunner


@pytest.fixture(scope='module')
def runner():
    """Runner for testing click CLIs"""
    return CliRunner()
