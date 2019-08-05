# -*- coding: utf-8 -*-
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
from concurrent.futures import ProcessPoolExecutor
import h5py
import os
import sys
import numpy as np
import pandas as pd
import logging
from warnings import warn
if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
# pylint: disable-msg=W0404
import matplotlib.pyplot as plt
import matplotlib as mpl

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

                    try:
                        scale1 = t.attrs1(dset)['psm_scale_factor']
                    except KeyError as _:
                        scale1 = 1
                        warn('Dataset "{}" does not have psm_scale_factor.'
                             .format(dset))

                    try:
                        scale2 = t.attrs2(dset)['psm_scale_factor']
                    except KeyError as _:
                        scale2 = 1
                        warn('Dataset "{}" does not have psm_scale_factor.'
                             .format(dset))

                    # make time-series dataframes with one site of data
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

    @staticmethod
    def multi_year(year_range, out_dir, dsets,
                   nsrdb_dir='/projects/PXS/nsrdb/v3.0.1/',
                   fname_base='nsrdb_{year}.h5',
                   timesteps=range(0, 17520, 8600), **kwargs):
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
            Spatial.dsets(h5, dsets, out_dir, timesteps=timesteps, **kwargs)

    @staticmethod
    def _fmt_title(kwargs, og_title, ti, ts, fname, dset, file_ext):
        """Format the figure title with timestamp."""

        if 'title' in kwargs:
            if og_title is None:
                og_title = kwargs['title']
            if '{}' in og_title:
                s = ('{}-{}-{} {}:{}'.format(
                    ti[ts].month, ti[ts].day, ti[ts].year,
                    str(ti[ts].hour).zfill(2),
                    str(ti[ts].minute).zfill(2)))
                kwargs['title'] = og_title.format(s)

        fname_out = '{}_{}_{}{}'.format(fname, dset, ts,
                                        file_ext)
        return fname_out, kwargs, og_title

    @staticmethod
    def dsets(h5, dsets, out_dir, timesteps=(0,), file_ext='.png',
              sites=None, interval=None, parallel=False, **kwargs):
        """Make map style plots at several timesteps for a given dataset.

        Parameters
        ----------
        h5 : str
            Target resource file with path.
        dsets : str | list
            Name of target dataset(s) to plot
        out_dir : str
            Path to dump output plot files.
        timesteps : iterable | slice
            Timesteps (time indices) to make plots for. Slice will have faster
            data extraction and will be later converted to an iterable.
        file_ext : str
            File extension
        sites : None | List | Slice
            sites to plot.
        interval : None | int
            Interval to plots sites at, i.e. if 100, only 1 every 100 sites
            will be plotted.
        parallel : bool
            Flag to generate plots in parallel (for each timestep).
        """

        og_title = None
        if isinstance(dsets, str):
            dsets = [dsets]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for dset in dsets:
            fname = os.path.basename(h5).replace('.h5', '')
            with h5py.File(h5, 'r') as f:
                logger.info('Plotting "{}" from {}.'.format(dset, h5))
                attrs = dict(f[dset].attrs)
                dset_shape = f[dset].shape

                ti = pd.to_datetime(f['time_index'][...].astype(str))
                if sites is None:
                    df = pd.DataFrame(f['meta'][...]).loc[:, ['latitude',
                                                              'longitude']]
                else:
                    df = pd.DataFrame(f['meta'][sites]).loc[:, ['latitude',
                                                                'longitude']]
            if interval is not None:
                df = df.iloc[slice(None, None, interval), :]

            if 'scale_factor' in attrs:
                scale_factor = attrs['scale_factor']
            elif 'psm_scale_factor' in attrs:
                scale_factor = attrs['psm_scale_factor']
            else:
                scale_factor = 1
                warn('Could not find scale factor attr in h5: {}'
                     .format(h5))

            # 2D array with timesteps
            if len(dset_shape) > 1:
                logger.debug('Importing data for "{}"...'.format(dset))
                with h5py.File(h5, 'r') as f:
                    if sites is None:
                        data = (f[dset][timesteps, :].astype(np.float32) /
                                scale_factor)
                    else:
                        data = (f[dset][timesteps, sites].astype(np.float32) /
                                scale_factor)
                if interval is not None:
                    data = data[:, slice(None, None, interval)]
                logger.debug('Finished importing data for "{}".'
                             .format(dset))

                if isinstance(timesteps, slice):
                    step = timesteps.step
                    if step is None:
                        step = 1
                    timesteps = list(range(timesteps.start,
                                           timesteps.stop,
                                           step))

                if not parallel:
                    for i, ts in enumerate(timesteps):
                        df[dset] = data[i, :]
                        fn_out, kwargs, og_title = Spatial._fmt_title(
                            kwargs, og_title, ti, ts, fname, dset,
                            file_ext)
                        Spatial.plot_geo_df(df, fn_out, out_dir,
                                            **kwargs)
                else:
                    with ProcessPoolExecutor() as exe:
                        for i, ts in enumerate(timesteps):
                            df_par = df.copy()
                            df_par[dset] = data[i, :]
                            fn_out, kwargs, og_title = Spatial._fmt_title(
                                kwargs, og_title, ti, ts, fname, dset,
                                file_ext)
                            exe.submit(Spatial.plot_geo_df, df_par, fn_out,
                                       out_dir, **kwargs)

            # 1D array, no timesteps
            else:
                with h5py.File(h5, 'r') as f:
                    if sites is None:
                        data = (f[dset][...].astype(np.float32) /
                                scale_factor)
                    else:
                        data = (f[dset][sites].astype(np.float32) /
                                scale_factor)

                if interval is not None:
                    data = data[slice(None, None, interval)]
                df[dset] = data
                fname_out = '{}_{}{}'.format(fname, dset, file_ext)
                Spatial.plot_geo_df(df, fname_out, out_dir, **kwargs)

    @staticmethod
    def goes_cloud(fpath, dsets, out_dir, nan_fill=-15, sparse_step=10,
                   **kwargs):
        """Plot datasets from a GOES cloud file.

        Parameters
        ----------
        fpath : str
            Target cloud file with path (either .nc or .h5).
        dsets : str | list
            Name of target dataset(s) to plot
        out_dir : str
            Path to dump output plot files.
        nan_fill : int | float
            Value to fill missing cloud data.
        sparse_step : int
            Step size to plot sites at a given interval (speeds things up).
            sparse_step=1 will plot all datapoints.
        """
        from nsrdb.data_model.clouds import (CloudVarSingleH5,
                                             CloudVarSingleNC,
                                             CloudVar)

        if fpath.endswith('.nc'):
            cld = CloudVarSingleNC(fpath, pre_proc_flag=True, index=None,
                                   dsets=dsets)
        else:
            cld = CloudVarSingleH5(fpath, pre_proc_flag=True, index=None,
                                   dsets=dsets)

        timestamp = CloudVar.get_timestamp(fpath)

        for dset in dsets:
            df = cld.grid.copy()
            data = cld.source_data[dset]
            data[np.isnan(data)] = nan_fill
            df[dset] = data

            df = df.iloc[slice(0, None, sparse_step)]

            fname = '{}_{}.png'.format(timestamp, dset)

            Spatial.plot_geo_df(df, fname, out_dir, **kwargs)

    @staticmethod
    def plot_geo_df(df, fname, out_dir, labels=('latitude', 'longitude'),
                    xlabel='Longitude', ylabel='Latitude', title=None,
                    cbar_label=None, marker_size=0.1,
                    xlim=(-127, -65), ylim=(24, 50), figsize=(10, 5),
                    cmap='OrRd_11', cbar_range=None, dpi=150,
                    extent=None, axis=None):
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
        marker_size : float
            Marker size.
            0.1 is good for CONUS at 4km.
        xlim : list | tuple
            Plot x limits (left limit, right limit).
        ylim : list | tuple
            Plot y limits (lower limit, upper limit).
        cmap : str
            Matplotlib colormap (Blues, OrRd)
        cbar_range = None | tuple
            Optional fixed range for the colormap.
        dpi : int
            Dots per inch.
        figsize : tuple
            Figure size inches (width, height).
        file_ext : str
            Image file extension (.png, .jpeg).
        """

        if isinstance(extent, str):
            if extent.lower() == 'conus':
                xlim = (-127, -65)
                ylim = (24, 50)
                figsize = (10, 5)
            elif extent.lower() == 'nsrdb':
                xlim = (-190, -20)
                ylim = (-62, 65)
                figsize = (10, 8)
            elif extent.lower() == 'canada':
                xlim = (-140, -50)
                ylim = (43, 68)
                figsize = (12, 7)
            elif extent.lower() == 'east':
                xlim = (-130, -20)
                ylim = (-62, 62)
                figsize = (7, 8)
            elif extent.lower() == 'wecc':
                xlim = (-127, -100)
                ylim = (29, 50)
                figsize = (8, 6)

        try:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            var = df.columns.values[2]

            if cbar_range is None:
                cbar_range = [np.nanmin(df.iloc[:, 2]),
                              np.nanmax(df.iloc[:, 2])]

            if '_' not in cmap:
                custom_cmap = False
                cmap = plt.get_cmap(cmap)

                c = ax.scatter(df.loc[:, labels[1]],
                               df.loc[:, labels[0]],
                               marker='s',
                               s=marker_size,
                               c=df.iloc[:, 2],
                               cmap=cmap,
                               vmin=cbar_range[0],
                               vmax=cbar_range[1])

            else:
                custom_cmap = True
                cmap_name, nbins = cmap.split('_')
                cmap = plt.get_cmap(cmap_name)
                cmaplist = [cmap(i) for i in range(cmap.N)]
                bounds = np.linspace(cbar_range[0], cbar_range[1], int(nbins))
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    '{}_{}'.format(cmap_name, nbins), cmaplist, len(bounds))
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                c = ax.scatter(df.loc[:, labels[1]],
                               df.loc[:, labels[0]],
                               marker='s',
                               s=marker_size,
                               c=df.iloc[:, 2],
                               cmap=cmap,
                               norm=norm)

            if xlabel is None:
                xlabel = labels[1]
            if ylabel is None:
                ylabel = labels[0]
            if title is None:
                title = fname
            if cbar_label is None:
                cbar_label = var

            if axis is not None:
                ax.axis(axis)
            else:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            if title is not False:
                ax.set_title(title)

            if not custom_cmap:
                fig.colorbar(c, ax=ax, label=cbar_label)
            else:
                fmt = '%.2f'
                int_bar = any([b % 1 == 0.0 for b in bounds])
                if int_bar:
                    fmt = '%.0f'
                fig.colorbar(c, ax=ax, label=cbar_label, cmap=cmap, norm=norm,
                             spacing='proportional',
                             ticks=bounds,
                             boundaries=bounds,
                             format=fmt)

            ax.set_ylim(ylim)
            ax.set_xlim(xlim)
            out = os.path.join(out_dir, fname)
            fig.savefig(out, dpi=dpi, bbox_inches='tight')
            logger.info('Saved figure: {}.png'.format(title))
            plt.close()
        except Exception as e:
            # never break a full data pipeline on failed plots
            logger.warning('Could not plot "{}". Received the following '
                           'exception: {}'.format(title, e))
