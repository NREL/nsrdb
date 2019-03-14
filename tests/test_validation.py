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
import pprint

from nsrdb.qa.statistics import mae, mae_perc, mbe, mbe_perc, rmse, rmse_perc
from nsrdb.all_sky.all_sky import all_sky


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


def get_at_interval(df, dt='5min', window=61):
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

    dhi_msr[dhi_msr == -9999] = np.nan
    dni_msr[dni_msr == -9999] = np.nan
    ghi_msr[ghi_msr == -9999] = np.nan

    measurement_df = pd.DataFrame({'dhi': dhi_msr, 'dni': dni_msr,
                                   'ghi': ghi_msr}, index=ti_msr)

    measurement_df = measurement_df.interpolate(method='linear', axis=0)

    measurement_df = measurement_df.sort_index()

    measurement_df = get_at_interval(measurement_df, dt='30min', window=61)

    return measurement_df


def get_source_data(test_file, sites=list(range(9))):
    """Retrieve the variables required to run all-sky for a given set of sites.
    """
    out = {}
    var_list = ('solar_zenith_angle', 'surface_pressure', 'surface_albedo',
                'ssa', 'aod', 'alpha', 'ozone', 'total_precipitable_water',
                'asymmetry')

    with h5py.File(test_file, 'r') as f:

        # get unscaled source variables
        out['time_index'] = pd.to_datetime(f['time_index'][...].astype(str))

        # get meta
        meta = pd.DataFrame(f['meta'][...])
        meta = meta.iloc[sites, :]
        print(meta[['state', 'county']])

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
    res = res.format(y=year)
    surfrad = surfrad.format(s=site_code, y=year)

    as_out, source_vars = run_all_sky(res, sites=[site], debug=False)

    nsrdb = pd.DataFrame({'dhi': as_out[0].flatten(),
                          'dni': as_out[1].flatten(),
                          'ghi': as_out[2].flatten(),
                          'cloud_type': source_vars['cloud_type'].flatten()},
                         index=source_vars['time_index'])

    measurement = get_measurement_data(surfrad)

    return nsrdb, measurement


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
    nsrdb, measurement = test_all_sky(site=7, year=2008)

    for var in ('dhi', 'dni', 'ghi'):
        plt.plot(nsrdb.index, nsrdb[var])
        plt.plot(measurement.index, measurement[var])
        plt.xlim([nsrdb.index[17000], nsrdb.index[17150]])
        plt.xlim([nsrdb.index[13000], nsrdb.index[13150]])
        plt.xlim([nsrdb.index[0], nsrdb.index[150]])
        plt.title(var)
        plt.legend(['NSRDB', 'Measurement'])
        plt.xticks(rotation=90)
        plt.show()
        plt.close()

    var_methods = {'mae': mae,
                   'mae_perc': mae_perc,
                   'mbe': mbe,
                   'mbe_perc': mbe_perc,
                   'rmse': rmse,
                   'rmse_perc': rmse_perc,
                   }

    stats = {}
    for k in var_methods.keys():
        stats[k] = {}

    for var in ('dhi', 'dni', 'ghi'):
        for k, method in var_methods.items():
            stats[k][var] = method(nsrdb[var].values, measurement[var].values)

    pprint.pprint(stats)
