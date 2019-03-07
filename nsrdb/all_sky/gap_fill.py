# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2018

@author: gbuster
"""

import pandas as pd
import numpy as np
from nsrdb.all_sky import CLOUD_TYPES


def make_fill_flag(irrad, cs_irrad, cloud_type, missing_cld_props):
    """Make a dataset indicating where to fill bad irradiance data.

    Parameters
    ----------
    irrad : np.ndarray
        All-sky (cloudy + clear) irradiance data.
    cs_irrad : np.ndarray
        Clearsky (REST) irradiance data.
    cloud_type : np.ndarray
        Array of numerical cloud types.
    missing_cld_props : np.ndarray
        Boolean array flagging timesteps with missing cloud properties.

    Returns
    -------
    fill_flag : np.ndarray
        Array of integers signifying whether to fill the irradiance
        data and the reason why. Fill flag codes:
            0 : No fill, irrad data is good
            1 : Missing cloud type (-15)
            2 : Cloudy but irradiance >= clearsky
            3 : Irradiance is NaN or < 0
            4 : Missing cloud properties
    """

    cloudy = np.isin(cloud_type, CLOUD_TYPES)

    # Make a categorical numerical fill flag
    fill_flag = np.zeros_like(irrad).astype(np.int8)
    fill_flag[(cloud_type == -15)] = 1
    fill_flag[(cloudy & (irrad >= cs_irrad))] = 2
    fill_flag[np.isnan(irrad) | (irrad < 0)] = 3
    fill_flag[missing_cld_props] = 4

    return fill_flag


def gap_fill_irrad(irrad, cs_irrad, fill_mask, return_csr=False):
    """Fill bad irradiance data using clearsky and nearest good cloudy data.

    Parameters
    ----------
    irrad : np.ndarray
        Full FARMS + REST2 merged irradiance 2D array.
    cs_irrad : np.ndarray
        REST2 clearsky irradiance without bad or missing data.
    fill_mask : np.ndarray
        2D boolean mask with True where bad data must be filled.
    return_csr : bool
        If set to true, this function will return the clearsky ratio array.

    Returns
    -------
    irrad : np.ndarray
        Updated and patched irradiance numpy array.
    csr : np.ndarray
        Irradiance cloudy/clearsky ratio numpy array.
    """
    # disable divide by zero warnings
    np.seterr(divide='ignore', invalid='ignore')

    # convert to boolean if necessary
    if np.issubdtype(fill_mask.dtype, np.integer):
        fill_mask = fill_mask.astype(bool)

    # get the merged div by clearsky ratio dataframe
    csr = irrad / cs_irrad

    # replace to-fill values with nan
    csr[fill_mask] = np.nan

    # fill nan ratio values with nearest good ratio values
    csr = pd.DataFrame(csr).interpolate(method='nearest', axis=0).\
        fillna(method='ffill').fillna(method='bfill').values

    # Set the cloud/clear ratio when it's nighttime
    csr[(cs_irrad == 0)] = 0

    # fill values with (closest good ratio values X
    # good clearsky data at the bad data index)
    irrad[fill_mask] = csr[fill_mask] * cs_irrad[fill_mask]

    if return_csr:
        return irrad, csr
    else:
        return irrad
