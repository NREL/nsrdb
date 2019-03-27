# pylint: skip-file
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os
import h5py
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nsrdb import TESTDATADIR
from nsrdb.qa.statistics import mbe_perc, rmse_perc
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.all_sky import CLEAR_TYPES, CLOUD_TYPES
from nsrdb.utilities.solar_position import SolarPosition


RTOL = 1e-03
ATOL = 0.001

# h5 index to surfrad site code name
SITE_CODES = {0: 'dra',  # desert rock nevada
              2: 'tbl',  # table mountain boulder
              3: 'fpk',  # fort peck Montana
              5: 'sxf',  # sioux falls South Dakota
              6: 'gwn',  # goodwin creek Mississippi
              7: 'bon',  # Bondville illinois
              8: 'psu'}  # Penn. State


def get_at_interval(df, dt, window=61):
    """Get a rolling avg dataset sampled at a given timestep interval."""
    year = df.index.year[0]
    for var in df:
        df[var] = df[var].rolling(window, center=True).mean()
        ti = pd.date_range('1-1-{y}'.format(y=year),
                           '1-1-{y}'.format(y=year + 1),
                           freq=dt)[:-1]
    df_new = pd.DataFrame(index=ti).join(df)
    return df_new


def get_measurement_data(surfrad_file):
    """Get original irradiance data from an NSRDB file to benchmark against.
    """

    with h5py.File(surfrad_file, 'r') as f:
        # get the measurement irradiance variables
        dhi_msr = f['dhi'][...].astype(float)
        dni_msr = f['dni'][...].astype(float)
        ghi_msr = f['ghi'][...].astype(float)
        ti_msr = pd.to_datetime(f['time_index'][...].astype(str))

    dhi_msr[dhi_msr < 0] = np.nan
    dni_msr[dni_msr < 0] = np.nan
    ghi_msr[ghi_msr < 0] = np.nan

    measurement_df = pd.DataFrame({'dhi': dhi_msr, 'dni': dni_msr,
                                   'ghi': ghi_msr}, index=ti_msr)

    measurement_df = measurement_df.sort_index()

    # calculate the window size to take a moving average over
    avg_min = 60  # take the average over this many minutes
    dt = ti_msr.to_series().diff()
    dt = dt[1].seconds / 60
    window = int(np.round(avg_min / dt)) + 1

    measurement_df = get_at_interval(measurement_df, dt='1h', window=window)

    return measurement_df


def get_source_data(test_file, sites=list(range(9))):
    """Retrieve the variables required to run all-sky for a given set of sites.
    """
    out = {}
    var_list = ('surface_pressure', 'surface_albedo',
                'ssa', 'aod', 'alpha', 'ozone', 'total_precipitable_water',
                'asymmetry')

    with h5py.File(test_file, 'r') as f:

        # get unscaled source variables
        out['time_index'] = pd.to_datetime(f['time_index'][...].astype(str))

        # get meta
        meta = pd.DataFrame(f['meta'][...])
        meta = meta.iloc[sites, :]
        print('Getting source data from "{}".'
              .format(str(meta['state'].values[0])))

        for var in var_list:
            out[var] = f[var][:, sites] / f[var].attrs['psm_scale_factor']

        # different unscaling for cloud properties
        out['cloud_type'] = f['cloud_type'][:, sites]
        out['cld_reff_dcomp'] = (
            f['cld_reff_dcomp'][:, sites].astype(float) *
            f['cld_reff_dcomp'].attrs['psm_scale_factor'] +
            f['cld_reff_dcomp'].attrs['psm_add_offset'])
        out['cld_opd_dcomp'] = (
            f['cld_opd_dcomp'][:, sites].astype(float) *
            f['cld_opd_dcomp'].attrs['psm_scale_factor'] +
            f['cld_opd_dcomp'].attrs['psm_add_offset'])

    out['solar_zenith_angle'] = SolarPosition(
        out['time_index'], meta[['latitude', 'longitude']].values).zenith

    return out


def run_all_sky(test_file, sites=list(range(9)), debug=False):
    """Run the all-sky processing code over the specified site list."""

    source_vars = get_source_data(test_file, sites=sites)

    # run all_sky processing
    all_sky_out = all_sky(**source_vars, debug=debug)
    return all_sky_out, source_vars


def plot_benchmark(sites, y_range=None):
    """Make plots benchmarking allsky irradiance against a baseline set of
    irradiances.
    """
    dhi, dni, ghi = run_all_sky(sites=sites, debug=False)

    ghi_orig, dni_orig, dhi_orig, fill_orig = get_measurement_data(sites=sites)

    # calculate maximum GHI differences and index locations
    ghi_diff = np.abs(ghi - ghi_orig)
    max_diff = np.max(ghi_diff, axis=0)
    loc_max_diff = np.argmax(ghi_diff, axis=0)

    for site in range(dhi.shape[1]):
        print('\nSite index {}'.format(site))

        if not y_range:
            # make center of x-axis the timestep of greatest error
            center = loc_max_diff[site]
            t0 = center - 50
            t1 = center + 50
        else:
            t0, t1 = y_range

        print('\nIndices {} through {} with diff {} at {}'
              .format(t0, t1, max_diff[site], loc_max_diff[site]))

        plt.hist(ghi_diff[:, site], bins=50)
        plt.ylim([0, 100])
        plt.xlabel('Error')
        plt.ylabel('Number of timesteps')
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), ghi_orig[t0:t1, site])
        plt.plot(range(t0, t1), ghi[t0:t1, site])
        plt.legend(['ghi baseline', 'ghi new'])
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), dni_orig[t0:t1, site])
        plt.plot(range(t0, t1), dni[t0:t1, site])
        plt.legend(['dni baseline', 'dni new'])
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), dhi_orig[t0:t1, site])
        plt.plot(range(t0, t1), dhi[t0:t1, site])
        plt.legend(['dhi baseline', 'dhi new'])
        plt.show()
        plt.close()


def test_all_sky(res='./data/validation_nsrdb/nsrdb_surfrad_{y}.h5',
                 surfrad='./data/validation_surfrad/{s}_{y}.h5',
                 site=0, year=1998, bad_threshold=0.005):
    """Run a numerical test of all_sky irradiance vs. benchmark NSRDB data."""

    site_code = SITE_CODES[site]
    res_file = res.format(y=year)
    surfrad_file = surfrad.format(s=site_code, y=year)

    if os.path.exists(surfrad_file):
        print('Running against {}'.format(surfrad_file))

        as_out, source_vars = run_all_sky(res_file, sites=[site], debug=False)

        nsrdb = pd.DataFrame(
            {'dhi': as_out[0].flatten(),
             'dni': as_out[1].flatten(),
             'ghi': as_out[2].flatten(),
             'cloud_type': source_vars['cloud_type'].flatten(),
             'sza': source_vars['solar_zenith_angle'].flatten()},
            index=source_vars['time_index'])

        # 1 hour filter for stats (take data at the top of the hour)
        nsrdb = nsrdb[(nsrdb.index.minute == 0)]

        measurement = get_measurement_data(surfrad_file)

        return nsrdb, measurement
    else:
        print('Skipping, does not exist: {}'.format(surfrad_file))
        return None


def calc_stats(nsrdb, measurement, stats=None, var_list=('dni', 'ghi'),
               flag='all', var_methods={'mbe_perc': mbe_perc,
                                        'rmse_perc': rmse_perc}):
    """Calculate the statistics for the NSRDB vs. Measurement irradiance."""

    if stats is None:
        stats = {}
        for k in var_methods.keys():
            stats[k] = {}
            for var in var_list:
                stats[k][var] = 0

    for var in var_list:
        for k, method in var_methods.items():

            if flag == 'all':
                # only take SZA < 80 degrees and positive irradiance
                mask = (nsrdb['sza'] < 80) & (nsrdb[var] > 0)

            elif flag == 'cloudy':
                mask = ((nsrdb['sza'] < 80) & (nsrdb[var] > 0) &
                        (nsrdb['cloud_type'].isin(CLOUD_TYPES)))

            elif flag == 'clear':
                mask = ((nsrdb['sza'] < 80) & (nsrdb[var] > 0) &
                        (nsrdb['cloud_type'].isin(CLEAR_TYPES)))

            nsrdb_vals = nsrdb[var][mask].values
            measu_vals = measurement[var][mask].values
            stats[k][var] += method(nsrdb_vals, measu_vals)

    return stats


def normalize_stats(stats, N):
    """Normalize (average) the stats over N samples. """
    for k1, v in stats.items():
        for k2 in v.keys():
            stats[k1][k2] = np.round(stats[k1][k2] / N, decimals=2)
    return stats


def stats_bar_chart(stats, var='dni', metric='mbe_perc', y_range=None,
                    figsize=(10, 5)):
    """Make a bar chart of NSRDB statistics."""

    COLORS = {"red": (0.7176, 0.1098, 0.1098),
              "green": (0.65 * 0.298, 0.65 * 0.6863, 0.65 * 0.3137),
              "blue": (0.9 * 0.0824, 0.9 * 0.3961, 0.9 * 0.7529),
              "orange": (0.85 * 1.0, 0.85 * 0.5961, 0.0),
              "purple": (0.49412, 0.3412, 0.7608),
              "grey": (0.45, 0.45, 0.45),
              "cyan": (0.0, 0.7373, 0.8314),
              "teal": (0.0, 0.5882, 0.5333),
              "lime": (0.8039, 0.8627, 0.2235),
              "brown": (0.4745, 0.3333, 0.2824),
              "black": (0.0, 0.0, 0.0)
              }
    f, ax = plt.subplots(figsize=figsize)
    n_bars = len(list(stats.keys()))

    barWidth = 1
    spacing = 1

    n_set = []
    in_set = []
    for i in range(n_bars):
        n_set.append(int(np.floor(i / 3)))
        in_set.append(i % 3)

    # The x position of bars
    r1 = np.arange(n_bars) * spacing + np.array(n_set)
    r2 = [x + barWidth for i, x in enumerate(r1)]

    color_list = list(COLORS.keys())

    for i, (k, v) in enumerate(stats.items()):
        plt.bar(r2[i], v.loc[var, metric], width=barWidth,
                color=COLORS[color_list[in_set[i]]],
                edgecolor='black', capsize=7, label=k)

    plt.legend(['All-Sky', 'Cloudy', 'Clear'])
    plt.xticks(np.array(r2), stats.keys(), rotation=90)
    plt.title('{} - {}'.format(var, metric))
    plt.ylabel(metric)
    plt.grid(axis='y')

    if y_range is not None:
        plt.ylim(y_range)

    fname = 'surfrad_validation_{}_{}.png'.format(var, metric)
    fout = os.path.join(TESTDATADIR, 'test_plots/', fname)
    plt.savefig(fout, bbox_inches='tight')
    print('Saved figure: "{}"'.format(fout))
    plt.close()


def execute_pytest(capture='all', flags='-rapP'):
    """Execute module as pytest with detailed summary report.

    Parameters
    ----------
    capture : str
        Log or stdout/stderr capture option. ex: log (only logger),
        all (includes stdout/stderr)
    flags : str
        Which tests to show logs and results for.
    """

    fname = os.path.basename(__file__)
    pytest.main(['-q', '--show-capture={}'.format(capture), fname, flags])


if __name__ == '__main__':

    out = {}
    for site in SITE_CODES.keys():
        print('Running for "{}"'.format(SITE_CODES[site]))
        stats = None
        cloudy_stats = None
        clear_stats = None
        N = 0
        for year in range(1998, 2018):
            result = test_all_sky(site=site, year=year)
            if result is not None:
                stats = calc_stats(result[0], result[1],
                                   stats=stats, flag='all')
                cloudy_stats = calc_stats(result[0], result[1],
                                          stats=cloudy_stats, flag='cloudy')
                clear_stats = calc_stats(result[0], result[1],
                                         stats=clear_stats, flag='clear')
                N += 1
        stats = normalize_stats(stats, N)
        cloudy_stats = normalize_stats(cloudy_stats, N)
        clear_stats = normalize_stats(clear_stats, N)

        df = pd.DataFrame(stats)
        cloudy_df = pd.DataFrame(cloudy_stats)
        clear_df = pd.DataFrame(clear_stats)

        out[SITE_CODES[site]] = df
        out[SITE_CODES[site] + '_cloudy'] = cloudy_df
        out[SITE_CODES[site] + '_clear'] = clear_df

    print('\n')
    for k, v in out.items():
        print(' ')
        print(k)
        print(v)

    # Save figures
    stats_bar_chart(out, var='dni', metric='mbe_perc', y_range=(-20, 20))
    stats_bar_chart(out, var='ghi', metric='mbe_perc', y_range=(-15, 15))

    stats_bar_chart(out, var='dni', metric='rmse_perc', y_range=(0, 100))
    stats_bar_chart(out, var='ghi', metric='rmse_perc', y_range=(0, 100))
