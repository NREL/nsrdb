"""PyTest file for main nsrdb CLI."""
import os
import tempfile
import traceback

import pytest
from click.testing import CliRunner

from nsrdb import TESTDATADIR, cli
from nsrdb.utilities.pytest import execute_pytest

pytest.importorskip("pyhdf")

BASE_DIR = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(BASE_DIR, './data/albedo')
TEST_MERRA_DIR = os.path.join(TESTDATADIR, 'merra2_source_files')


@pytest.fixture(scope="module")
def runner():
    """Runner for testing click CLIs"""
    return CliRunner()


def test_cli_create_configs(runner):
    """Test nsrdb.cli create-configs"""
    with tempfile.TemporaryDirectory() as td:
        kwargs = {'year': 2020,
                  'outdir': td}
        result = runner.invoke(cli.create_configs, ['-kw', kwargs])

        # assert result.exit_code == 0
        if result.exit_code != 0:
            msg = ('Failed with error {}'
                   .format(traceback.print_exception(*result.exc_info)))
            raise RuntimeError(msg)


if __name__ == '__main__':
    execute_pytest(__file__)
