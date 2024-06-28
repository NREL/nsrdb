"""Pytest helper utilities."""

import os

import pytest


def execute_pytest(file, capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    file : str
        Name of pytest file to run
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(file)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])
