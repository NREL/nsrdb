# -*- coding: utf-8 -*-
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
import h5py
import os
import pandas as pd
import logging

from nsrdb.utilities.loggers import init_logger


logger = logging.getLogger(__name__)


def plot_multi_year(year_range, out_dir, dsets,
                    nsrdb_dir='/projects/PXS/nsrdb/v3.0.1/',
                    fname_base='nsrdb_{year}.h5',
                    timesteps=range(0, 17520, 8600)):
    """Make map plots at timesteps for datasets in multiple NSRDB files.

    Parameters
    ----------
    year_range : iterable
        List of integer or string values to plot data for.
    out_dir : str
        Path to dump output plot files.
    dsets : str | list
        Name of target dataset(s) to plot
    nsrdb_dir : str
        Target resource file directory.
    fname_base : str
        Base nsrdb filename found in the nsrdb_dir. Should have a {year}
        keyword.
    timesteps : iterable
        Timesteps (time indices) to make plots for.
    """

    for year in year_range:
        h5 = os.path.join(nsrdb_dir, fname_base.format(year=year))
        plot_dset(h5, dsets, out_dir, timesteps=timesteps)


def plot_dset(h5, dsets, out_dir, timesteps=range(0, 17520, 8600)):
    """Make map style plots at several timesteps for a given dataset.

    Parameters
    ----------
    h5 : str
        Target resource file with path.
    dsets : str | list
        Name of target dataset(s) to plot
    out_dir : str
        Path to dump output plot files.
    timesteps : iterable
        Timesteps (time indices) to make plots for.
    """

    init_logger(__name__, log_file=None, log_level='INFO')

    if isinstance(dsets, str):
        dsets = [dsets]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dset in dsets:
        with h5py.File(h5, 'r') as f:
            logger.info('Plotting "{}" from {}.'.format(dset, h5))
            fname = os.path.basename(h5).replace('.h5', '')
            df = pd.DataFrame(f['meta'][...]).loc[:, ['latitude', 'longitude']]
            attrs = dict(f[dset].attrs)

            for i in timesteps:
                logger.info('Plotting timestep {}'.format(i))
                df[dset] = f[dset][i, :] / attrs.get('psm_Scale_factor', 1)
                plot_geo_df(df, fname + '_' + dset + '_{}'.format(i), out_dir)


def plot_geo_df(df, title, out_dir, labels=('latitude', 'longitude'),
                xlim=(-190, -20), ylim=(-30, 70)):
    """Plot a dataframe to verify the blending operation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with latitude/longitude in 1st/2nd columns, data in the
        3rd column.
    title : str
        Figure and file title.
    out_dir : str
        Where to save the plot.
    labels : list | tuple
        latitude/longitude column labels.
    xlim : list | tuple
        Plot x limits (left limit, right limit). (-190, -20) is whole NSRDB.
    ylim : list | tuple
        Plot y limits (lower limit, upper limit). (-30, 70) is whole NSRDB.
    """

    try:
        # HPC matplot lib import
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cmap = plt.get_cmap('Blues')

        cbar_range = [df.iloc[:, 2].min(), df.iloc[:, 2].max()]

        var = df.columns.values[2]

        c = ax.scatter(df.loc[:, labels[1]],
                       df.loc[:, labels[0]],
                       marker='s',
                       s=0.5,
                       c=df.iloc[:, 2],
                       cmap=cmap,
                       vmin=cbar_range[0],
                       vmax=cbar_range[1])
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        fig.colorbar(c, ax=ax, label=var)
        ax.set_title(title)
        out = os.path.join(out_dir, title + '.png')
        fig.savefig(out, dpi=600)
        logger.info('Saved figure: {}.png'.format(title))
        plt.close()
    except Exception as e:
        logger.warning('Could not plot "{}". Received the following '
                       'exception: {}'.format(title, e))
