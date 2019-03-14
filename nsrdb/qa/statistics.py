# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:29:51 2018

@author: gbuster
"""
import numpy as np


def mae(x, x_true):
    """Calculate the mean absolute error (has units)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    diff = np.abs(x - x_true)
    sum_diff = np.sum(diff)
    mae = sum_diff / len(x)
    return mae


def mae_perc(x, x_true):
    """Calculate the mean absolute error (percentage of true)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    mae_val = mae(x, x_true)
    mae_perc = 100 * mae_val / (np.abs(np.mean(x_true)))
    return mae_perc


def mbe(x, x_true):
    """Calculate the mean bias error (has units)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    diff = x - x_true
    sum_diff = np.sum(diff)
    mbe = sum_diff / len(x)
    return mbe


def mbe_perc(x, x_true):
    """Calculate the mean bias error (percentage of true)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    mbe_val = mbe(x, x_true)
    mbe_perc = 100 * mbe_val / (np.mean(x_true))
    return mbe_perc


def rmse(x, x_true):
    """Calculate the root mean square error (has units)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    diff2 = (x - x_true) ** 2
    sum_diff2 = np.sum(diff2)
    rmse = (sum_diff2 / len(x)) ** 0.5
    return rmse


def rmse_perc(x, x_true):
    """Calculate the root mean square error (percentage of true)."""
    x = np.nan_to_num(x)
    x_true = np.nan_to_num(x_true)
    rmse_val = rmse(x, x_true)
    denom = np.mean(x_true ** 2) ** 0.5
    rmse_perc = 100 * rmse_val / denom
    return rmse_perc
