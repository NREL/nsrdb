# -*- coding: utf-8 -*-
# flake8: noqa: C901
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
import datetime
import h5py
import logging
import numpy as np
import os
import pandas as pd
import sys
from warnings import warn

if 'linux' in sys.platform:
    import matplotlib
    matplotlib.use('Agg')
# pylint: disable-msg=W0404
import matplotlib.pyplot as plt
import matplotlib as mpl

from rex.utilities.loggers import init_logger
from rex.utilities.execution import SpawnProcessPool

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
        self._gids1 = None
        self._gids2 = None
        self._meta1 = None
        self._meta2 = None
        self._t1 = None
        self._t2 = None
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

        if self._meta1 is None:
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

        if self._meta2 is None:
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

        if self._t1 is None:
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

        if self._t2 is None:
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
                    except KeyError:
                        scale1 = 1
                        warn('Dataset "{}" does not have psm_scale_factor.'
                             .format(dset))

                    try:
                        scale2 = t.attrs2(dset)['psm_scale_factor']
                    except KeyError:
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
                    loc_check = all(np.round(loc1.astype(float),
                                             decimals=2)
                                    == np.round(loc2.astype(float),
                                                decimals=2))
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
                            title = (dset
                                     + '_{}_{}_{}'.format(site1, month, day))
                            t.plot_timeseries(df2, df1, title, dset, legend,
                                              out_dir, month=month, day=day)


class Spatial:
    """Framework to perform NSRDB spatial QA via map plots."""

    EXTENTS = {'conus': {'xlim': (-127, -65),
                         'ylim': (13, 50),
                         'figsize': (10, 6)},
               'nsrdb': {'xlim': (-190, -20),
                         'ylim': (-23, 61),
                         'figsize': (10, 6)},
               'canada': {'xlim': (-140, -50),
                          'ylim': (43, 68),
                          'figsize': (12, 7)},
               'east': {'xlim': (-130, -20),
                        'ylim': (-62, 62),
                        'figsize': (7, 8)},
               'west': {'xlim': (-180, -100),
                        'ylim': (-60, 62),
                        'figsize': (7, 8)},
               'full': {'xlim': (-170, -20),
                        'ylim': (-62, 62),
                        'figsize': (10, 7)},
               'wecc': {'xlim': (-127, -100),
                        'ylim': (29, 50),
                        'figsize': (8, 6)},
               'south_america': {'xlim': (-85, -32),
                                 'ylim': (-59, 16),
                                 'figsize': (7, 9)},
               'global': {'xlim': (-180, 180),
                          'ylim': (-90, 90),
                          'figsize': (120, 80)},
               'meteosat': {'xlim': (-24, 108),
                            'ylim': (-54, 60),
                            'figsize': (10, 7)},
               'himawari': {'xlim': (55, 180),
                            'ylim': (-60, 60),
                            'figsize': (9, 7)},
               }

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

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Plotting figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            Plotting axes object.
        """

        init_logger(__name__, log_file=None, log_level='INFO')

        for year in year_range:
            h5 = os.path.join(nsrdb_dir, fname_base.format(year=year))
            fig, ax = Spatial.dsets(h5, dsets, out_dir, timesteps=timesteps,
                                    **kwargs)
        return fig, ax

    @staticmethod
    def _fmt_title(kwargs, og_title, ti, ts, fname, dset, file_ext, timedelta):
        """Format the figure title with timestamp."""

        if timedelta is not None:
            ti += timedelta

        if 'title' in kwargs and kwargs['title']:
            if og_title is None:
                og_title = kwargs['title']

            s = ('{}-{}-{} {}:{}'.format(
                ti[ts].month, ti[ts].day, ti[ts].year,
                str(ti[ts].hour).zfill(2),
                str(ti[ts].minute).zfill(2)))
            if '{}' in og_title:
                kwargs['title'] = og_title.format(s)
            if '{}' in fname:
                fname = fname.format(s.replace(':', '-'))

        fname_out = '{}_{}_{}{}'.format(fname, dset, ts,
                                        file_ext)
        return fname_out, kwargs, og_title

    @staticmethod
    def dsets(h5, dsets, out_dir, timesteps=(0,), fname=None, file_ext='.png',
              sites=None, interval=None, max_workers=1,
              timedelta=datetime.timedelta(hours=0),
              **kwargs):
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
        fname : str
            Filename without extension.
        file_ext : str
            File extension
        sites : None | List | Slice
            sites to plot.
        interval : None | int
            Interval to plots sites at, i.e. if 100, only 1 every 100 sites
            will be plotted.
        max_workers : int | None
            Maximum number of workers to use (>1 is parallel).
        timedelta : datetime.timedelta
            Time delta object to shift the time index printed inthe title
            if {} is present in the title. This has no effect on the input
            timesteps.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Plotting figure object (Only returned if not parallel).
        ax : matplotlib.axes._subplots.AxesSubplot
            Plotting axes object (Only returned if not parallel).
        """

        og_title = None
        if isinstance(dsets, str):
            dsets = [dsets]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if fname is None:
            fname = os.path.basename(h5).replace('.h5', '')

        for dset in dsets:
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
                        data = (f[dset][timesteps, :].astype(np.float32)
                                / scale_factor)
                    else:
                        data = (f[dset][timesteps, sites].astype(np.float32)
                                / scale_factor)
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

                if max_workers == 1:
                    for i, ts in enumerate(timesteps):
                        df[dset] = data[i, :]
                        fn_out, kwargs, og_title = Spatial._fmt_title(
                            kwargs, og_title, ti, ts, fname, dset,
                            file_ext, timedelta)
                        fig, ax = Spatial.plot_geo_df(df, fn_out, out_dir,
                                                      **kwargs)
                else:
                    fig, ax = None, None
                    with SpawnProcessPool(loggers='nsrdb',
                                          max_workers=max_workers) as exe:
                        for i, ts in enumerate(timesteps):
                            df_par = df.copy()
                            df_par[dset] = data[i, :]
                            fn_out, kwargs, og_title = Spatial._fmt_title(
                                kwargs, og_title, ti, ts, fname, dset,
                                file_ext, timedelta)
                            exe.submit(Spatial.plot_geo_df, df_par, fn_out,
                                       out_dir, **kwargs)

            # 1D array, no timesteps
            else:
                with h5py.File(h5, 'r') as f:
                    if sites is None:
                        data = (f[dset][...].astype(np.float32)
                                / scale_factor)
                    else:
                        data = (f[dset][sites].astype(np.float32)
                                / scale_factor)

                if interval is not None:
                    data = data[slice(None, None, interval)]
                df[dset] = data
                fname_out = '{}_{}{}'.format(fname, dset, file_ext)
                fig, ax = Spatial.plot_geo_df(df, fname_out, out_dir, **kwargs)

        return fig, ax

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

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Plotting figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            Plotting axes object.
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

            fig, ax = Spatial.plot_geo_df(df, fname, out_dir, **kwargs)

        return fig, ax

    @staticmethod
    def plot_geo_df(df, fname, out_dir, labels=('latitude', 'longitude'),
                    xlabel='Longitude', ylabel='Latitude',
                    title=None, title_loc='center',
                    cbar_label='dset', marker_size=0.1, marker='s',
                    xlim=(-127, -65), ylim=(24, 50), figsize=(10, 5),
                    cmap='OrRd_11', cbar_range=None, dpi=150,
                    extent=None, axis=None, alpha=1.0,
                    shape=None, shape_aspect=None,
                    shape_edge_color=(0.2, 0.2, 0.2), shape_line_width=2,
                    bbox_inches='tight', dark=True):
        """Plot a dataframe to verify the blending operation.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with latitude/longitude in 1st/2nd columns, data in the
            3rd column.
        fname : str
            Filename to save to including file extension (usually png).
        out_dir : str
            Output directory to save fname output.
        labels : list | tuple
            latitude/longitude column labels in the df.
        xlabel : str
            Label to put on the xaxis in the plot. Disabled if axis='off'
        ylabel : str
            Label to put on the yaxis in the plot. Disabled if axis='off'
        title : str
            Figure and file title.
        title_loc : str
            Location of title (center, left, right).
        cbar_label : str
            Label for the colorbar. Should usually include units of the plot.
        marker_size : float
            Marker size.
            0.1 is good for CONUS at 4km.
        marker : string
            marker shape 's' is square marker.
        xlim : list | tuple
            Plot x limits (left limit, right limit) (longitude bounds).
        ylim : list | tuple
            Plot y limits (lower limit, upper limit) (latitude bounds).
        figsize : tuple
            Figure size inches (width, height).
        cmap : str
            Matplotlib colormap (Blues, OrRd, viridis)
        cbar_range = None | tuple
            Optional fixed range for the colormap.
        dpi : int
            Dots per inch.
        extent : str
            Optional extent kwarg to fix the axis bounds and figure size
            from class.EXTENTS
        axis : None | str
            Option to turn axis "off"
        alpha : float
            Transparency value between 0 (transparent) and 1 (opaque)
        shape : str
            Filepath to a shape file to plot on top of df data (only the
            boundaries are plotted).
        shape_aspect : None | float
            Optional aspect ratio to use on shape.
        shape_edge_color : str | tuple
            Color of the plotted shape edges
        shape_line_width : float
            Line width of the shape edges
        bbox_inches : str
            kwarg for saving figure.
        dark : bool
            Flag to turn on dark mode plots.

        Returns
        -------
        fig : matplotlib.pyplot.figure
            Plotting figure object.
        ax : matplotlib.axes._subplots.AxesSubplot
            Plotting axes object.
        """

        df = df.sort_values(by=list(labels))
        textcolor = 'k'
        facecolor = None
        if dark:
            textcolor = '#969696'
            facecolor = 'k'

        if isinstance(extent, str):
            if extent.lower() in Spatial.EXTENTS:
                xlim = Spatial.EXTENTS[extent.lower()]['xlim']
                ylim = Spatial.EXTENTS[extent.lower()]['ylim']
                figsize = Spatial.EXTENTS[extent.lower()]['figsize']

        try:
            fig = plt.figure(figsize=figsize, facecolor=facecolor)
            ax = fig.add_subplot(111)
            var = df.columns.values[2]
            fig.patch.set_facecolor(facecolor)
            mpl.rcParams['text.color'] = textcolor
            mpl.rcParams['axes.labelcolor'] = textcolor

            if cbar_range is None:
                cbar_range = [np.nanmin(df.iloc[:, 2]),
                              np.nanmax(df.iloc[:, 2])]
            elif isinstance(cbar_range, tuple):
                cbar_range = list(cbar_range)
            if cbar_range[0] is None:
                cbar_range[0] = np.nanmin(df.iloc[:, 2])
            if cbar_range[1] is None:
                cbar_range[1] = np.nanmax(df.iloc[:, 2])

            if '_' not in cmap:
                custom_cmap = False
                cmap = plt.get_cmap(cmap)

                # hack for colorbar if alpha is input
                c = ax.scatter(df.iloc[0][labels[1]],
                               df.iloc[0][labels[0]],
                               c=df.iloc[0, 2],
                               cmap=cmap,
                               vmin=cbar_range[0],
                               vmax=cbar_range[1],
                               alpha=1.0)
                _ = ax.scatter(df.loc[:, labels[1]],
                               df.loc[:, labels[0]],
                               marker=marker,
                               s=marker_size,
                               c=df.iloc[:, 2],
                               cmap=cmap,
                               vmin=cbar_range[0],
                               vmax=cbar_range[1],
                               alpha=alpha)

            else:
                custom_cmap = True
                nbins = cmap.split('_')[-1]
                cmap_name = cmap.replace('_' + nbins, '')
                cmap = plt.get_cmap(cmap_name)
                cmaplist = [cmap(i) for i in range(cmap.N)]
                bounds = np.linspace(cbar_range[0], cbar_range[1], int(nbins))
                cmap = mpl.colors.LinearSegmentedColormap.from_list(
                    '{}_{}'.format(cmap_name, nbins), cmaplist, len(bounds))
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

                # hack for colorbar if alpha is input
                c = ax.scatter(df.iloc[0][labels[1]],
                               df.iloc[0][labels[0]],
                               marker=marker,
                               s=marker_size,
                               c=df.iloc[0, 2],
                               cmap=cmap,
                               norm=norm,
                               alpha=1.0)
                _ = ax.scatter(df.loc[:, labels[1]],
                               df.loc[:, labels[0]],
                               marker=marker,
                               s=marker_size,
                               c=df.iloc[:, 2],
                               cmap=cmap,
                               norm=norm,
                               alpha=alpha)

            if shape is not None:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame.from_file(shape)
                gdf = gdf.to_crs({'init': 'epsg:4326'})
                gdf.geometry.boundary.plot(ax=ax, color=None,
                                           edgecolor=shape_edge_color,
                                           linewidth=shape_line_width)
                if shape_aspect:
                    ax.set_aspect(shape_aspect)

            if xlabel is None:
                xlabel = labels[1]
            if ylabel is None:
                ylabel = labels[0]
            if title is None:
                title = fname
            if cbar_label == 'dset':
                cbar_label = var

            if title is not False:
                ax.set_title(title, loc=title_loc, color=textcolor)

            cbar = None
            if not custom_cmap and cbar_label is not None:
                cbar = fig.colorbar(c, ax=ax)
            elif cbar_label is not None:
                fmt = '%.2f'
                int_bar = all(b % 1 == 0.0 for b in bounds)
                if int_bar:
                    fmt = '%.0f'
                cbar = fig.colorbar(c, ax=ax, cmap=cmap,
                                    norm=norm,
                                    spacing='proportional',
                                    ticks=bounds,
                                    boundaries=bounds,
                                    format=fmt)

            if cbar is not None:
                ticks = plt.getp(cbar.ax, 'yticklabels')
                plt.setp(ticks, color=textcolor)
                cbar.ax.tick_params(which='minor', color=textcolor)
                cbar.ax.tick_params(which='major', color=textcolor)
                cbar.set_label(cbar_label, color=textcolor)

            if ylim is not None:
                ax.set_ylim(ylim)
            if xlim is not None:
                ax.set_xlim(xlim)

            if axis is not None:
                bounds = ax.axis(axis)
                logger.info('Axes bounds are: {}'.format(bounds))
            else:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            out = os.path.join(out_dir, fname)
            if fname.endswith('.tiff'):
                from PIL import Image
                import io
                png1 = io.BytesIO()
                fig.savefig(png1, format='png', dpi=dpi, bbox_inches='tight',
                            facecolor=facecolor)
                png2 = Image.open(png1)
                png2.save(out)
                png2.close()
            else:
                fig.savefig(out, dpi=dpi, bbox_inches=bbox_inches,
                            facecolor=facecolor)
            logger.info('Saved figure: {}'.format(fname))
            plt.close()
        except Exception as e:
            # never break a full data pipeline on failed plots
            msg = ('Could not plot "{}". Received the following '
                   'exception: {}'.format(title, e))
            logger.error(msg)
            raise e

        return fig, ax
