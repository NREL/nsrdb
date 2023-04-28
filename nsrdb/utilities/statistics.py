# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:29:51 2018

@author: gbuster
"""
import numpy as np


def N(x, x_true):
    """Get the number of non-nan values common to both arrays."""
    return np.sum(~(np.isnan(x) | np.isnan(x_true)))


def mae(x, x_true):
    """Calculate the mean absolute error (has units)."""
    diff = np.abs(x - x_true)
    sum_diff = np.nansum(diff)
    mae_out = sum_diff / N(x, x_true)

    return mae_out


def mae_perc(x, x_true):
    """Calculate the mean absolute error (percentage of true)."""
    mae_val = mae(x, x_true)
    mae_perc_out = 100 * mae_val / (np.abs(np.nanmean(x_true)))

    return mae_perc_out


def mbe(x, x_true):
    """Calculate the mean bias error (has units)."""
    diff = x - x_true
    sum_diff = np.nansum(diff)
    mbe_out = sum_diff / N(x, x_true)

    return mbe_out


def mbe_perc(x, x_true):
    """Calculate the mean bias error (percentage of true)."""
    mbe_val = mbe(x, x_true)
    mbe_perc_out = 100 * mbe_val / (np.nanmean(x_true))

    return mbe_perc_out


def rmse(x, x_true):
    """Calculate the root mean square error (has units)."""
    diff2 = (x - x_true) ** 2
    sum_diff2 = np.nansum(diff2)
    rmse_out = (sum_diff2 / N(x, x_true)) ** 0.5

    return rmse_out


def rmse_perc(x, x_true):
    """Calculate the root mean square error (percentage of true)."""
    rmse_val = rmse(x, x_true)
    denom = np.nanmean(x_true ** 2) ** 0.5
    rmse_perc_out = 100 * rmse_val / denom

    return rmse_perc_out
