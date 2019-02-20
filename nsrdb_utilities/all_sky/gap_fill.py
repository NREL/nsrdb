# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2018

@author: gbuster
"""

import numpy as np


def gap_fill_ghi(ghi, cs_ghi, fill_mask, return_csr=False):
    """Fill bad data in GHI array using clearsky and nearest good cloudy data.

    Parameters
    ----------
    ghi : pd.DataFrame
        Full FARMS+REST2 merged GHI dataframe.
    cs_ghi : pd.DataFrame
        REST2 clearsky GHI without bad or missing data.
    fill_mask : pd.DataFrame
        2D boolean mask with True where bad data must be filled.
    return_csr : bool
        If set to true, this function will return the clearsky ratio array.

    Returns
    -------
    ghi : pd.DataFrame
        Updated and patched ghi dataframe.
    cr : pd.DataFrame
        GHI cloudy/clearsky ratio dataframe.
    """

    # get the merged div by clearsky ratio dataframe
    cr = ghi / cs_ghi

    # replace to-fill values with nan
    cr.values[fill_mask] = np.nan

    # Set the cloud/clear ratio to zero when it's nighttime
    night_mask = (cs_ghi == 0)
    cr.values[night_mask] = 0

    # fill nan ratio values with nearest good ratio values
    cr = cr.interpolate(method='nearest', axis=0).fillna(method='ffill')

    # fill values with (closest good ratio values X
    # good clearsky data at the bad data index)
    ghi.values[fill_mask] = cr.values[fill_mask] * cs_ghi.values[fill_mask]

    if return_csr:
        return ghi, cr
    else:
        return ghi
