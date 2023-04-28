# -*- coding: utf-8 -*-
"""NSRDB interpolation algorithms for downscaling.

Created on Tue Dec  4 08:22:26 2018

@author: gbuster
"""
import logging
import numpy as np
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree

logger = logging.getLogger(__name__)


def to_rads(locs):
    """Convert angle array to radians"""
    return locs * (np.pi / 180.0)


def rad_to_dist(locs):
    """Multiply radians by earth's radius to get distance"""
    R = 6373.0  # radius of the earth in kilometers
    return locs * R


def reg_grid_nn(latitude, longitude, df, labels=('latitude', 'longitude')):
    """Get nearest neighbors from a regular lat/lon grid.

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

    Returns
    -------
    out : ndarray
        Array of indices that correspond to data mapped to the regular
        latitude/longitude input grid. The regular grid data should be
        flattened and then indexed with this output indices var.
    """

    # make a kdTree for lat and lon seperately
    if len(latitude.shape) == 1:
        latitude = np.expand_dims(latitude, axis=1)
    if len(longitude.shape) == 1:
        longitude = np.expand_dims(longitude, axis=1)

    logger.debug('Building lat/lon cKDTrees with lengths {} and {} '
                 'respectively for a regular grid.'
                 .format(len(latitude), len(longitude)))
    # pylint: disable=not-callable
    lat_tree = cKDTree(latitude)
    lon_tree = cKDTree(longitude)

    logger.debug('Querying lat/lon cKDTrees with lengths {} and {} '
                 'respectively for a regular grid.'
                 .format(len(latitude), len(longitude)))
    # query each tree with the df lats/lons seperately
    _, lat_ind = lat_tree.query(
        np.expand_dims(df[labels[0]].values, axis=1), k=1)
    _, lon_ind = lon_tree.query(
        np.expand_dims(df[labels[1]].values, axis=1), k=1)

    # convert corresponding 2D arrays to flattened index
    out = np.ravel_multi_index([lat_ind, lon_ind],
                               (len(latitude), len(longitude)))
    # pylint: disable=no-member
    out = out.astype(np.uint32)

    return out


def knn(df1, df2, labels=('latitude', 'longitude'), k=1):
    """Get nearest neighbors for data sets.

    Parameters
    ----------
    df1/df2 : pd.DataFrame:
        Dataframes containing columns with the corresponding labels.
    labels : tuple | list
        Column labels corresponding to the columns in df1/df2.
    k : int
        Number of nearest neighbors to return

    Returns
    -------
    dist : ndarray
        Distance array in decimal degrees.
    indicies : ndarray
        2D array of row indicies in df1 that match df2.
        df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
    """

    # need list for pandas column indexing
    if not isinstance(labels, list):
        labels = list(labels)

    logger.debug('Building cKDTrees for {} coordinates.'
                 .format(len(df1)))
    # pylint: disable=not-callable
    tree = cKDTree(df1[labels].values)

    logger.debug('Querying cKDTrees for {} coordinates.'
                 .format(len(df2)))
    dist, ind = tree.query(df2[labels].values, k=k)

    dist = dist.astype(np.float32)
    ind = ind.astype(np.uint32)

    if len(dist.shape) == 1:
        dist = np.expand_dims(dist, axis=1)
        ind = np.expand_dims(ind, axis=1)

    return dist, ind


def geo_nn(df1, df2, labels=('latitude', 'longitude'), k=4):
    """Get geo nearest neighbors for coordinate sets using haversine dist.

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
        Distance array in km.
    indicies : ndarray
        2D array of row indicies in df1 that match df2.
        df1[df1.index[indicies[i]]] is closest to df2[df2.index[i]]
    """

    # need list for pandas column indexing
    if not isinstance(labels, list):
        labels = list(labels)

    coords1 = to_rads(df1[labels].values)
    coords2 = to_rads(df2[labels].values)

    logger.debug('Building Haversine BallTree for {} coordinates.'
                 .format(len(df1)))
    tree = BallTree(coords1, metric='haversine')

    logger.debug('Querying Haversine BallTree for {} coordinates.'
                 .format(len(df2)))
    dist, ind = tree.query(coords2, return_distance=True, k=k)

    dist = rad_to_dist(dist)

    dist = dist.astype(np.float32)
    ind = ind.astype(np.uint32)

    if len(dist.shape) == 1:
        dist = np.expand_dims(dist, axis=1)
        ind = np.expand_dims(ind, axis=1)

    return dist, ind
