# -*- coding: utf-8 -*-
"""NSRDB interpolation algorithms for downscaling.

Created on Tue Dec  4 08:22:26 2018

@author: gbuster
"""
import logging
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree


logger = logging.getLogger(__name__)


def temporal_lin(array, ti_native, ti_new):
    """Linearly interpolate an array at a native timeindex to the new timeindex

    Parameters
    ----------
    array : np.ndarray
        Data (time X space) at native temporal resolution.
    ti_native : pd.DatetimeIndex
        Pandas datetime index for the array native (old) resolution.
    ti_new : pd.DatetimeIndex
        Pandas datetime index for the array desired (new) resolution.

    Returns
    -------
    array : np.ndarray
        Data at new temporal resolution
    """

    array = pd.DataFrame(array, index=ti_native).reindex(ti_new)\
        .interpolate(method='linear', axis=0).values

    return array


def temporal_step(array, ti_native, ti_new):
    """Stepwise interpolate an array at a native timeindex to the new timeindex

    Parameters
    ----------
    array : np.ndarray
        Data (time X space) at native temporal resolution.
    ti_native : pd.DatetimeIndex
        Pandas datetime index for the array native (old) resolution.
    ti_new : pd.DatetimeIndex
        Pandas datetime index for the array desired (new) resolution.

    Returns
    -------
    array : np.ndarray
        Data at new temporal resolution
    """

    if array.shape[0] > 1:
        array = pd.DataFrame(array, index=ti_native).reindex(ti_new)\
            .interpolate(method='nearest', axis=0)\
            .fillna(method='ffill').fillna(method='bfill').values
    else:
        # single entry arrays cannot be interpolated but must be filled
        array = pd.DataFrame(array, index=ti_native).reindex(ti_new)\
            .fillna(method='ffill').fillna(method='bfill').values

    return array


def to_rads(locs):
    """Convert angle array to radians"""
    return locs * (np.pi / 180.0)


def rad_to_dist(locs):
    """Multiply radians by earth's radius to get distance"""
    R = 6373.0  # radius of the earth in kilometers
    return locs * R


def reg_grid_nn(latitude, longitude, df, labels=('latitude', 'longitude'),
                k=4):
    """Get nearest neighbors for coordinate sets.

    Parameters
    ----------
    latitude : np.ndarray
        1D array of latitude values from a regular grid.
    longitude : np.ndarray
        1D array of longitude values from a regular grid.
    df : pd.DataFrame:
        Dataframe containing coodinate columns with the corresponding
        labels.
    labels : tuple | list
        Column labels corresponding to the lat/lon columns in df.
    k : int
        Number of nearest neighbors to return

    Returns
    -------
    out : ndarray
        Array of indices that correspond to data mapped to the regular
        latitude/longitude input grid. The regular grid data should be
        flattened and then indexed with this output indices var.
    """

    # make a kdTree for lat and lon seperately
    lat_tree = cKDTree(latitude)
    lon_tree = cKDTree(longitude)

    # query each tree with the df lats/lons seperately
    _, lat_ind = lat_tree.query(df[labels[0]].values, k=k)
    _, lon_ind = lon_tree.query(df[labels[1]].values, k=k)

    # initialize the output array
    out = np.ndarray((len(df), k), dtype=np.uint32)

    for i in range(k):
        # make 2D arrays corresponding the lat/lon index results
        mesh_lon_ind, mesh_lat_ind = np.meshgrid(lon_ind[:, i], lat_ind[:, i])
        mesh_lat_ind = mesh_lat_ind.astype(np.uint32)
        mesh_lon_ind = mesh_lon_ind.astype(np.uint32)

        # convert corresponding 2D arrays to flattened index
        one_d_index = np.ravel_multi_index(
            [mesh_lat_ind.ravel(), mesh_lon_ind.ravel()],
            (len(latitude), len(longitude)))
        one_d_index = one_d_index.astype(np.uint32)

        # assign to output array based on k
        out[:, i] = one_d_index

    return out


def geo_nn(df1, df2, labels=('latitude', 'longitude'), k=4):
    """Get nearest neighbors for coordinate sets.

    Parameters
    ----------
    df1/df2 : pd.DataFrame:
        Dataframes containing coodinate columns with the corresponding labels.
    labels : tuple | list
        Column labels corresponding to the lat/lon columns in df1/df2.
    k : int
        Number of nearest neighbors to return

    Returns
    -------
    dist : ndarray
        Distance array in km returned if return_dist input arg set to True.
    indicies : ndarray
        1D array of row indicies in df1 that match df2.
        df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
    """

    # need list for pandas column indexing
    if not isinstance(labels, list):
        labels = list(labels)

    coords1 = to_rads(np.array(df1[labels]))
    coords2 = to_rads(np.array(df2[labels]))
    tree = BallTree(coords1, metric='haversine')
    dist, ind = tree.query(coords2, return_distance=True, k=k)
    dist = rad_to_dist(dist)
    dist = dist.astype(np.float32)
    ind = ind.astype(np.uint32)
    return dist, ind


def agg(data, indices):
    """Spatal interpolation by aggregation of nearest neighbors.

    Parameters
    ----------
    data : np.array
        Coarse resolution NSRDB data to be interpolated.
        Must be for a single timestep and be flattened/raveled.
    indices : np.ndarray
        Index array generated by a sklearn Tree query.

    Returns
    -------
    out : np.array
        NSRDB data that has been aggregated.
    """

    # calculate the aggregated data
    data_agg = np.mean(data[indices], axis=1)

    return data_agg


def idw(data, indices, distance, p=2, dist_thresh=None):
    """Inverse Distance Weighted Spatial Interpolation.

    Parameters
    ----------
    data : np.array
        Coarse resolution NSRDB data to be interpolated.
        Must be for a single timestep and be flattened/raveled.
    indices : np.ndarray
        Index array generated by a sklearn Tree query.
    distance : np.ndarray
        Distance array generated by a sklearn Tree query.
    p : int
        Power of the inverse distance. Default = 2.
    dist_thresh : int | float
        Distance threshold. Neighbors with distance greater than this value
        are not interpolated.

    Returns
    -------
    out : np.array
        Fine resolution NSRDB data that has been interpolated.
    """

    weight = 1.0 / np.power(distance, p)

    # Need to account for divide by zero which is result of perfect overlap.
    # Note numpy doesn't doesn't bomb on divide by zero, it instead returns NaN
    # this assigns the zero-dist index with 100% weighting, 0% to other indices
    rows = np.where(distance == 0)[0]
    weight[rows, 0] = 1
    weight[rows, 1:] = 0

    # calculate the IDW
    data_idw = (np.sum(weight * data[indices], axis=1) /
                np.sum(weight, axis=1))

    if dist_thresh is not None:
        data_idw = np.where(distance[:, 0] < dist_thresh, data_idw, np.nan)
    return data_idw


def nn(data, indices, distance=None, dist_thresh=None):
    """Nearest Neighbor Spatial Interpolation.

    Parameters
    -------------
    data : np.array
        Coarse resolution NSRDB data to be interpolated.
        Must be for a single timestep and be flattened/raveled.
    indices : np.ndarray
        Index array generated by a sklearn Tree query.
    distance : np.ndarray
        Optional distance array generated by a sklearn Tree query for distance
        filtering.
    dist_thresh : int | float
        Distance threshold. Neighbors with distance greater than this value
        are not interpolated.

    Returns
    -------
    out : np.array
        Fine resolution NSRDB data that has been interpolated.
    """
    data_nn = data[indices]
    if distance is not None and dist_thresh is not None:
        data_nn = np.where(distance < dist_thresh, data_nn, np.nan)
    return data_nn


def lower_data(var, data, var_elev):
    """Lower Data Values using Elevation and Scale Height

    scale height 2500 meters (per Chris email dated 12/29/2013 11:12pm)

    Parameters
    ----------
    var : str
        NSRDB variable name.
    data : np.array
        NSRDB variable data. Must be for a single timestep and be
        flattened/raveled.
    var_elev : np.ndarray
        Elevation data for the native variable resolution
        (usually elevation for the coarse grid).

    Returns
    -------
    data : np.array
        NSRDB variable data lowered based on native elevation.
    """

    if var == 'air_temperature':
        # air temp must be in C or K
        const = 6.5 / 1000.
        data = data + var_elev * const

    elif var == 'surface_pressure':
        # surface pressure should be scaled in Pascals
        if np.max(data) > 10000:
            # imply Pa
            scalar = 101325
        else:
            # imply mbar
            scalar = 1013.25
        const = scalar * (1 - (np.power((1 - var_elev / 44307.69231),
                                        5.25328)))
        data = data + const
        if np.min(data) < 0.0:
            raise ValueError('Spatial interpolation of surface pressure '
                             'resulted in negative values. Incorrectly '
                             'scaled/unscaled values or incorrect units are '
                             'the most likely causes.')

    elif var in ['aod', 'total_precipitable_water']:
        scale_height = 2500
        elev_scale = np.exp(var_elev / scale_height)
        data = data * elev_scale

    return data


def raise_data(var, data, ref_elev):
    """Raise Data Values using Elevation and Scale Height

    scale height 2500 meters (per Chris email dated 12/29/2013 11:12pm)

    Parameters
    ----------
    var : str
        NSRDB variable name.
    data : np.array
        NSRDB variable data. Must be for a single timestep and be
        flattened/raveled.
    ref_elev : np.ndarray
        Elevation data for the reference grid
        (usually elevation for the high-res grid).

    Returns
    -------
    data : np.array
        NSRDB variable data raised based on NSRDB reference elevation.
    """

    if var == 'air_temperature':
        # air temp must be in C or K
        const = 6.5 / 1000.
        data = data - ref_elev * const

    elif var == 'surface_pressure':
        # surface pressure should be scaled in Pascals
        if np.max(data) > 10000:
            # imply Pa
            scalar = 101325
        else:
            # imply mbar
            scalar = 1013.25
        const = scalar * (1 - (np.power((1 - ref_elev / 44307.69231),
                                        5.25328)))
        data = data - const
        if np.min(data) < 0.0:
            raise ValueError('Spatial interpolation of surface pressure '
                             'resulted in negative values. Incorrectly '
                             'scaled/unscaled values or incorrect units are '
                             'the most likely causes.')

    elif var in ['aod', 'total_precipitable_water']:
        scale_height = 2500
        elev_scale = np.exp(-ref_elev / scale_height)
        data = data * elev_scale

    return data


def parse_method(method):
    """Parse a method string for a numeric option as the last char.

    Parameters
    ----------
    method : str
        Interpolation method string. Can have a numeric option as last char.

    Returns
    -------
    option : int | None
        Numeric option if present in method, or None if no option is found.
    """

    try:
        option = int(method[-1])
        return option
    except ValueError as _:
        return None


def spatial_interp(var, data, native_grid, new_grid, method, dist, ind,
                   elevation_correct):
    """Perform single variable spatial interpolation on MERRA data array.

    Parameters
    ----------
    var : str
        NSRDB variable name.
    data : np.ndarray
        2D data array (time X sites) for a single variable.
    native_grid : pd.DataFrame
        Native grid associated with the input data.
        Should have column for elevation.
    new_grid : pd.DataFrame
        New grid associated with the output data.
        Should have column for elevation.
    method : str
        Interpolation method (IDW for inverse distance weighted, NN for
        nearest neighbor, AGG for aggregation). Last character can be an
        integer to set IDW power or number of AGG neighbors.
    dist : ndarray
        Distance array in km returned if return_dist input arg set to True.
    indicies : ndarray
        1D array of row indicies in native_grid that match new_grid.
    elevation_correct : bool
        Flag to perform elevation correction.

    Returns
    -------
    out_array : np.ndarray
        2D data array (time X sites) with sites matching the
        new_grid resolution.
    """

    # initialize output array
    out_array = np.zeros(shape=(len(data), len(new_grid)), dtype=np.float32)

    # iterate through timesteps interpolating all sites for a single timestep
    for i in range(data.shape[0]):

        # flatten the data for processing
        sub_data = data[i, :].ravel()

        # optional elevation correction
        if elevation_correct:
            sub_data = lower_data(var, sub_data,
                                  native_grid['elevation'].values)

        # perform interpolation on flat data using indices and distance inputs
        if 'IDW' in method.upper():
            p = parse_method(method)
            if p is None:
                p = 2
            sub_data = idw(sub_data, ind, dist, p=p)

        if 'AGG' in method.upper():
            sub_data = agg(sub_data, ind)

        elif 'NN' in method.upper():
            sub_data = nn(sub_data, ind, dist)

        # optional elevation correction
        if elevation_correct:
            sub_data = raise_data(var, sub_data,
                                  new_grid['elevation'].values)

        # ensure data is flat (ideally from shape of (n, 1) to (n,))
        if len(sub_data.shape) > 1:
            if sub_data.shape[1] != 1:
                raise ValueError('Spatial interpolation for a single timestep '
                                 'returned non 1D data of shape: {}'
                                 .format(sub_data.shape))
            sub_data = sub_data.ravel()

        # save single timestep of high spatial resolutiond data
        out_array[i, :] = sub_data

    return out_array
