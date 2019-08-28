# -*- coding: utf-8 -*-
"""NSRDB aggregation methods (higher spatiotemporal resolution to 4km 30min).
"""
import numpy as np
import pandas as pd
from nsrdb.utilities.plots import Spatial


def meta_source_2018(fpath_4km):
    """Make 2018 4km meta data with data source column (west/east/conus).

    WARNING: This is a script very specific to the 2018 GOES data arrangement,
    with the 4km 30min GOES West satellite, the 2km 15min GOES East satellite,
    and the 2km 5min CONUS data from GOES East. This only works with the psm v3
    4km meta data (accessed on 8/28/2019).

    Parameters
    ----------
    fpath_4km : str
        File path to full 4km meta data.

    Returns
    -------
    meta : pd.DataFrame
        DataFrame based on fpath_4km but with a "source" column containing
        the data source string.
    """

    meta = pd.read_csv(fpath_4km, index_col=0)
    meta['source'] = 'west'

    # east 2km longitude boundary is at -125 lon (just west of CONUS)
    east_mask = (meta.longitude > -125.0)
    meta.loc[east_mask, 'source'] = 'east'

    # conus includes all of US except for Alaska and Hawaii
    conus_mask = ((meta.country == 'United States')
                  & ~meta.state.isin(['Alaska', 'Hawaii']))
    meta.loc[conus_mask, 'source'] = 'conus'

    # made a line specific to the observed 2018 GOES East extreme angle
    # boundary, above which no cloud properties are returned for the East data.
    lat_boundary = 0.6 * (meta.longitude.values + 125) + 42.7
    angle_mask = ((meta.latitude > lat_boundary)
                  & (meta.source != 'west')
                  & (meta.longitude < -104.5))
    meta.loc[angle_mask, 'source'] = 'west'

    return meta


def plot_meta_source(fpath_4km, fname, out_dir, **kwargs):
    """Make a map plot of the NSRDB Meta source data (west/east/conus).

    Parameters
    ----------
    fpath_4km : str
        File path to full 4km meta data.
    fname : str
        Filename for output map image file.
    out_dir : str
        Directory path to save map plot file.
    **kwargs : dict
        Keyword args for spatial plotting utility.
    """

    meta = meta_source_2018(fpath_4km)
    sources = list(set(meta.source.unique()))
    meta['isource'] = np.nan
    for i, source in enumerate(sources):
        meta.loc[(meta.source == source), 'isource'] = i

    meta = meta[['latitude', 'longitude', 'isource']]
    Spatial.plot_geo_df(meta, fname, out_dir, **kwargs)


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
