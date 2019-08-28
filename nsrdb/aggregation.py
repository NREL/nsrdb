# -*- coding: utf-8 -*-
"""NSRDB aggregation methods (higher spatiotemporal resolution to 4km 30min).
"""
import numpy as np
import pandas as pd


def time_avg(inp, window=15):
    """Calculate the rolling time average for an input array or df.

    Parameters
    ----------
    inp : np.ndarray | pd.DataFrame
        Input array/df with data to average.
    window : int
        Window over which to calculate the average.

    Returns
    -------
    out : np.ndarray | pd.DataFrame
        Array or dataframe with same size as input and each value is a moving
        average.
    """

    array = False
    if isinstance(inp, np.ndarray):
        array = True
        inp = pd.DataFrame(inp)

    out = inp.rolling(window, center=True, min_periods=1).mean()

    if array:
        out = out.values

    return out
