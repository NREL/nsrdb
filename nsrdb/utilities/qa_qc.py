# -*- coding: utf-8 -*-
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
import h5py
import os
import sys
import numpy as np
import pandas as pd
import logging
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
# pylint: disable-msg=W0404
import matplotlib.pyplot as plt

from nsrdb.utilities.loggers import init_logger


logger = logging.getLogger(__name__)


class Temporal:
    """Framework to perform NSRDB temporal QA via timeseries plots."""

    def __init__(self, f1, f2, gids1=None, gids2=None):
        """
        Parameters
        ----------
        f1/f2 : str
            Two NSRDB h5 files to benchmark against each other (with paths).
        gids1/gids2 : NoneType | str | list
            Optional file-specific GID's to assign to each h5 file. None will
            default to the index in the h5 meta data. A string is interpreted
            as a meta data file with the first column being the GIDs.
        """

        logger.info('Performing temporal QA of {} and {}'
                    .format(os.path.basename(f1), os.path.basename(f2)))
        self._h1 = h5py.File(f1, 'r')
        self._h2 = h5py.File(f2, 'r')
        self.gids1 = gids1
        self.gids2 = gids2

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self._h1.close()
        self._h2.close()

        if type is not None:
            raise

    @property
    def gids1(self):
        """Get the gid list for h5 file 1.

        Returns
        -------
        _gids1 : list
            List of gids specific for h5 file 1.
        """
        return self._gids1

    @gids1.setter
    def gids1(self, inp):
        """Set the gid's for h5 file #1.

        Parameters
        ----------
        inp : NoneType | str | list
            Optional file-specific GID's to assign to h5 file #1. None will
            default to the index in the h5 meta data. A string is interpreted
            as a meta data file with the first column being the GIDs.
        """

        if isinstance(inp, str):
            self._gids1 = list(pd.read_csv(inp, index_col=0).index)
        elif inp is None:
            self._gids1 = list(self.meta1.index)
        else:
            self._gids1 = list(inp)

    @property
    def gids2(self):
        """Get the gid list for h5 file 2.

        Returns
        -------
        _gids2 : list
            List of gids specific for h5 file 2.
        """
        return self._gids2

    @gids2.setter
    def gids2(self, inp):
        """Set the gid's for h5 file #2.

        Parameters
        ----------
        inp : NoneType | str | list
            Optional file-specific GID's to assign to h5 file #2. None will
            default to the index in the h5 meta data. A string is interpreted
            as a meta data file with the first column being the GIDs.
        """

        if isinstance(inp, str):
            self._gids2 = list(pd.read_csv(inp, index_col=0).index)
        elif inp is None:
            self._gids2 = list(self.meta2.index)
        else:
            self._gids2 = list(inp)

    def attrs1(self, dset):
        """Get an attributes dictionary associated with dset in h1."""
        return dict(self.h1[dset].attrs)

    def attrs2(self, dset):
        """Get an attributes dictionary associated with dset in h1."""
        return dict(self.h2[dset].attrs)

    @property
    def h1(self):
        """Get the h5 file #1 h5py file handler."""
        return self._h1

    @property
    def h2(self):
        """Get the h5 file #2 h5py file handler."""
        return self._h2

    @property
    def meta1(self):
        """Get the meta data for file 1.

        Returns
        -------
        _meta1 : pd.DataFrame
            DataFrame representation of the 'meta' dataset in h1.
        """

        if not hasattr(self, '_meta1'):
            self._meta1 = pd.DataFrame(self.h1['meta'][...])
        return self._meta1

    @property
    def meta2(self):
        """Get the meta data for file 2.

        Returns
        -------
        _meta1 : pd.DataFrame
            DataFrame representation of the 'meta' dataset in h2.
        """

        if not hasattr(self, '_meta2'):
            self._meta2 = pd.DataFrame(self.h2['meta'][...])
        return self._meta2

    @property
    def t1(self):
        """Get the time index for file 1.

        Returns
        -------
        _t1 : pd.DateTime
            Get the timeseries index for h1 in pandas DateTime format.
        """

        if not hasattr(self, '_t1'):
            self._t1 = pd.to_datetime(self.h1['time_index'][...].astype(str))
        return self._t1

    @property
    def t2(self):
        """Get the time index for file 2.

        Returns
        -------
        _t2 : pd.DateTime
            Get the timeseries index for h2 in pandas DateTime format.
        """

        if not hasattr(self, '_t2'):
            self._t2 = pd.to_datetime(self.h2['time_index'][...].astype(str))
        return self._t2

    @staticmethod
    def plot_timeseries(df1, df2, title, ylabel, legend, out_dir,
                        month=1, day=1):
        """Plot a single day timeseries for two timeseries-indexed dataframes.

        Parameters
        ----------
        df1/df2 : pd.DataFrame
            Data from these two dataframes are plotted against each other.
            Each dataframe must have a pandas datetime index with day/month
            attributes. Only data in the 1st column (index 0) is plotted.
        title : str
            Plot title and output filename.
        ylabel : str
            Label for the y-axis.
        legend : list | tuple
            Two-entry legend specification.
        out_dir : str
            Location to dump plot image files.
        month : int
            Month of the time index in df1/df2 to plot.
        day : int
            Day of the time index in df1/df2 to plot.
        """

        mask1 = (df1.index.month == month) & (df1.index.day == day)
        mask2 = (df2.index.month == month) & (df2.index.day == day)

        plt.plot(df1.index[mask1], df1.iloc[mask1, 0], '-o')
        plt.plot(df2.index[mask2], df2.iloc[mask2, 0], '--x')

        y_min = np.min((np.min(df1.iloc[mask1, 0]),
                        np.min(df2.iloc[mask2, 0])))
        y_max = np.max((np.max(df1.iloc[mask1, 0]),
                        np.max(df2.iloc[mask2, 0])))
        plt.ylim((y_min, 1.1 * y_max))

        plt.xlabel('Time Index')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(legend)
        plt.xticks(rotation=90)

        plt.savefig(os.path.join(out_dir, title + '.png'), dpi=300,
                    bbox_inches='tight')
        plt.close()

    @classmethod
    def plot_sites(cls, f1, f2, gids1=None, gids2=None, dsets=('dni',),
                   sites=(0,), months=(1,), days=(1,), out_dir='./'):
        """Plot sites in file1 against a baseline file2.

        Parameters
        ----------
        f1/f2 : str
            Two NSRDB h5 files to benchmark against each other (with paths).
            f2 is interpreted as the baseline full NSRDB file.
        gids1/gids2 : NoneType | str | list
            Optional file-specific GID's to assign to each h5 file. None will
            default to the index in the h5 meta data. A string is interpreted
            as a meta data file with the first column being the GIDs.
        dsets : iterable
            Dataset names (str) to plot from each file.
        sites : iterable
            Site indices to plot. This will be interpreted as the index in f1,
            with gids1 used to find the index in f2 (if the indices do not
            match between f1/f2).
        months : iterable
            Months to plot.
        days : iterable
            Days to plot.
        out_dir : str
            Directory to dump output plot image files.
        """

        init_logger(__name__, log_file=None, log_level='INFO')
        legend = (os.path.basename(f2), os.path.basename(f1))
        with cls(f1, f2, gids1=gids1, gids2=gids2) as t:
            for i in sites:
                site1 = i
                site2 = t.gids1[i]

                for dset in dsets:
                    logger.info('Plotting dataset "{}"'.format(dset))

                    # make time-series dataframes with one site of data
                    scale1 = t.attrs1(dset).get('psm_scale_factor', 1)
                    scale2 = t.attrs2(dset).get('psm_scale_factor', 1)
                    df1 = pd.DataFrame({dset: t.h1[dset][:, site1] / scale1},
                                       index=t.t1)
                    df2 = pd.DataFrame({dset: t.h2[dset][:, site2] / scale2},
                                       index=t.t2)

                    # check that the locations match
                    loc1 = t.meta1.loc[site1, ['latitude', 'longitude']].values
                    loc2 = t.meta2.loc[site2, ['latitude', 'longitude']].values
                    loc_check = all(np.round(loc1.astype(float), decimals=2) ==
                                    np.round(loc2.astype(float), decimals=2))
                    if not loc_check:
                        logger.warning('Temporal QA sites do not match. '
                                       'Site in file 1 has index {} and '
                                       'lat/lon {}, site in file 2 has '
                                       'index {} and lat/lon {}'
                                       .format(site1, loc1, site2, loc2))
                    else:
                        logger.info('Plotting timeseries for site index {} in '
                                    'file 1 and site index {} in file 2 with '
                                    'lat/lon {}'.format(site1, site2, loc1))

                    for month in months:
                        for day in days:
                            title = (dset +
                                     '_{}_{}_{}'.format(site1, month, day))
                            t.plot_timeseries(df2, df1, title, dset, legend,
                                              out_dir, month=month, day=day)


class Spatial:
    """Framework to perform NSRDB spatial QA via map plots."""

    def multi_year(self, year_range, out_dir, dsets,
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

        init_logger(__name__, log_file=None, log_level='INFO')

        for year in year_range:
            h5 = os.path.join(nsrdb_dir, fname_base.format(year=year))
            self.dsets(h5, dsets, out_dir, timesteps=timesteps)

    def dsets(self, h5, dsets, out_dir, timesteps=range(0, 17520, 8600)):
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

        if isinstance(dsets, str):
            dsets = [dsets]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for dset in dsets:
            with h5py.File(h5, 'r') as f:
                logger.info('Plotting "{}" from {}.'.format(dset, h5))
                fname = os.path.basename(h5).replace('.h5', '')
                df = pd.DataFrame(f['meta'][...]).loc[:, ['latitude',
                                                          'longitude']]
                attrs = dict(f[dset].attrs)

                for i in timesteps:
                    logger.info('Plotting timestep {}'.format(i))
                    df[dset] = (f[dset][i, :] /
                                attrs.get('psm_Scale_factor', 1))
                    self.plot_geo_df(df, fname + '_' + dset + '_{}'.format(i),
                                     out_dir)

    @staticmethod
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
            Plot x limits (left limit, right limit). (-190, -20) is whole NSRDB
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
