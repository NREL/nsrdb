# pylint: skip-file
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nsrdb import TESTDATADIR
from nsrdb.all_sky.all_sky import all_sky
from nsrdb.utilities.pytest import execute_pytest
from nsrdb.utilities.statistics import mae_perc

TEST_FILE = os.path.join(
    TESTDATADIR, 'validation_nsrdb_farmsdni', 'nsrdb_2000.h5'
)
RTOL = 1e-03
ATOL = 0.01

SITES = list(range(10))


def get_benchmark_data(test_file=TEST_FILE, sites=SITES):
    """Get original irradiance data from an NSRDB file to benchmark against."""

    with h5py.File(test_file, 'r') as f:
        # get the original baseline irradiance variables
        dhi_orig = f['dhi'][:, sites]
        dni_orig = f['dni'][:, sites]
        ghi_orig = f['ghi'][:, sites]
        fill_orig = f['fill_flag'][:, sites]

    return dhi_orig, dni_orig, ghi_orig, fill_orig


def get_source_data(test_file=TEST_FILE, sites=SITES):
    """Retrieve the variables required to run all-sky for a given set of
    sites."""
    out = {}
    var_list = (
        'solar_zenith_angle',
        'surface_pressure',
        'surface_albedo',
        'ssa',
        'aod',
        'alpha',
        'ozone',
        'total_precipitable_water',
        'asymmetry',
    )

    with h5py.File(test_file, 'r') as f:
        # get unscaled source variables
        out['time_index'] = pd.to_datetime(f['time_index'][...].astype(str))

        for var in var_list:
            out[var] = f[var][:, sites] / f[var].attrs['psm_scale_factor']

        # different unscaling for cloud properties
        out['cloud_type'] = f['cloud_type'][:, sites]
        out['cld_reff_dcomp'] = (
            f['cld_reff_dcomp'][:, sites].astype(float)
            / f['cld_reff_dcomp'].attrs['psm_scale_factor']
        )
        out['cld_opd_dcomp'] = (
            f['cld_opd_dcomp'][:, sites]
            / f['cld_opd_dcomp'].attrs['psm_scale_factor']
        )

    return out


def run_all_sky(test_file=TEST_FILE, sites=SITES):
    """Run the all-sky processing code over the specified site list."""

    source_vars = get_source_data(test_file=test_file, sites=sites)

    # run all_sky processing
    all_sky_out = all_sky(**source_vars)
    return all_sky_out


def make_df(site):
    """Get dataframes containing single-site timeseries data for checking."""
    if isinstance(site, int):
        site = [site]

    d = get_source_data(sites=site)

    aso = run_all_sky(sites=site)

    dhi_orig, dni_orig, ghi_orig, fill_orig = get_benchmark_data(sites=site)

    df_dhi = pd.DataFrame(
        {
            'ti': d['time_index'],
            'sza': d['solar_zenith_angle'].flatten(),
            'cloud_type': d['cloud_type'].flatten(),
            'fill_flag': aso['fill_flag'].flatten(),
            'fill_flag_orig': fill_orig.flatten(),
            'cld_opd': d['cld_opd_dcomp'].flatten(),
            'dhi': aso['dhi'].flatten(),
            'dhi_orig': dhi_orig.flatten(),
            'cs_dhi': aso['clearsky_dhi'].flatten(),
        }
    )
    df_dni = pd.DataFrame(
        {
            'ti': d['time_index'],
            'sza': d['solar_zenith_angle'].flatten(),
            'cloud_type': d['cloud_type'].flatten(),
            'fill_flag': aso['fill_flag'].flatten(),
            'fill_flag_orig': fill_orig.flatten(),
            'cld_opd': d['cld_opd_dcomp'].flatten(),
            'dni': aso['dni'].flatten(),
            'dni_orig': dni_orig.flatten(),
            'cs_dni': aso['clearsky_dni'].flatten(),
        }
    )
    df_ghi = pd.DataFrame(
        {
            'ti': d['time_index'],
            'sza': d['solar_zenith_angle'].flatten(),
            'cloud_type': d['cloud_type'].flatten(),
            'fill_flag': aso['fill_flag'].flatten(),
            'fill_flag_orig': fill_orig.flatten(),
            'cld_opd': d['cld_opd_dcomp'].flatten(),
            'ghi': aso['ghi'].flatten(),
            'ghi_orig': ghi_orig.flatten(),
            'cs_ghi': aso['clearsky_ghi'].flatten(),
        }
    )

    return df_dhi, df_dni, df_ghi


def plot_benchmark(sites, y_range=None):
    """Make plots benchmarking allsky irradiance against a baseline set of
    irradiances.
    """
    aso = run_all_sky(sites=sites)

    dhi_orig, dni_orig, ghi_orig, _ = get_benchmark_data(sites=sites)

    # calculate maximum GHI differences and index locations
    ghi_diff = np.abs(aso['ghi'] - ghi_orig)
    max_diff = np.max(ghi_diff, axis=0)
    loc_max_diff = np.argmax(ghi_diff, axis=0)

    for site in range(aso['dhi'].shape[1]):
        print('\nSite index {}'.format(site))

        if not y_range:
            # make center of x-axis the timestep of greatest error
            center = loc_max_diff[site]
            t0 = center - 50
            t1 = center + 50
        else:
            t0, t1 = y_range

        print(
            '\nIndices {} through {} with diff {} at {}'.format(
                t0, t1, max_diff[site], loc_max_diff[site]
            )
        )

        plt.hist(ghi_diff[:, site], bins=50)
        plt.ylim([0, 100])
        plt.xlabel('Error')
        plt.ylabel('Number of timesteps')
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), ghi_orig[t0:t1, site])
        plt.plot(range(t0, t1), aso['ghi'][t0:t1, site])
        plt.legend(['ghi baseline', 'ghi new'])
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), dni_orig[t0:t1, site])
        plt.plot(range(t0, t1), aso['dni'][t0:t1, site])
        plt.legend(['dni baseline', 'dni new'])
        plt.show()
        plt.close()

        plt.plot(range(t0, t1), dhi_orig[t0:t1, site])
        plt.plot(range(t0, t1), aso['dhi'][t0:t1, site])
        plt.legend(['dhi baseline', 'dhi new'])
        plt.show()
        plt.close()


def test_all_sky(
    test_file=TEST_FILE,
    sites=SITES,
    timestep_frac_threshold=0.15,
    mae_perc_threshold=5,
):
    """Run a numerical test of all_sky irradiance vs. benchmark NSRDB data."""

    baseline = {}

    new = run_all_sky(test_file=test_file, sites=sites)

    baseline_results = get_benchmark_data(test_file=test_file, sites=sites)
    baseline['dhi'] = baseline_results[0]
    baseline['dni'] = baseline_results[1]
    baseline['ghi'] = baseline_results[2]

    max_perc_bad = 0

    mae_p = {'dhi': 0, 'dni': 0, 'ghi': 0}

    for i, site in enumerate(sites):
        for var in ('dhi', 'dni', 'ghi'):
            hist, _ = np.histogram(
                np.abs(new[var][:, i] - baseline[var][:, i]),
                bins=100,
                range=(0.0, 1000.0),
            )

            n_bad = np.sum(hist[2:])
            frac_bad = n_bad / new['ghi'].shape[0]
            max_perc_bad = np.max((max_perc_bad, 100 * frac_bad))

            msg = (
                '{:.4f}% of the values do not match the baseline '
                'irradiance (threshold is {:.4f}%) for site {}.'.format(
                    100 * frac_bad, 100 * timestep_frac_threshold, site
                )
            )
            assert frac_bad < timestep_frac_threshold, msg

            mae_p[var] += mae_perc(new[var][:, i], baseline[var][:, i])

    mae_p['dhi'] = np.round(mae_p['dhi'] / len(sites), decimals=2)
    mae_p['dni'] = np.round(mae_p['dni'] / len(sites), decimals=2)
    mae_p['ghi'] = np.round(mae_p['ghi'] / len(sites), decimals=2)

    for var in ('dhi', 'dni', 'ghi'):
        msg = 'Mean absolute error for "{}" in "{}" is {}%'.format(
            var, test_file, mae_p[var]
        )
        assert mae_p[var] < mae_perc_threshold, msg

    print(mae_p)
    print(
        'Maximum of {:.4f}% bad timesteps. Threshold was {:.4f}%.'.format(
            max_perc_bad, 100 * timestep_frac_threshold
        )
    )


def iter_speed_compare(sites=SITES):
    """Run a speed test comparing broadcasted all-sky to iterated all-sky."""

    t0 = time.time()
    run_all_sky(sites=sites)
    t_broad = time.time() - t0

    t0 = time.time()
    for site in sites:
        run_all_sky(sites=[site])
    t_iter = time.time() - t0

    print(
        'Running {} sites through all sky took {:.2f} seconds '
        'broadcasted, {:.2f} seconds iterated.'.format(
            len(sites), t_broad, t_iter
        )
    )


if __name__ == '__main__':
    execute_pytest(__file__)
