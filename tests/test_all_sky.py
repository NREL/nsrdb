# pylint: skip-file
"""
PyTest file for rest2.

Created on Feb 13th 2019

@author: gbuster
"""

import os
import h5py
import pytest
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nsrdb.all_sky.all_sky import all_sky


TEST_FILE = './data/test_nsrdb_2017.h5'
RTOL = 1e-03
ATOL = 0.001


def get_benchmark_data(test_file=TEST_FILE, sites=list(range(10))):
    """Get original irradiance data from an NSRDB file to benchmark against.
    """

    with h5py.File(test_file, 'r') as f:

        # get the original baseline irradiance variables
        ghi_orig = f['ghi'][:, sites]
        dni_orig = f['dni'][:, sites]
        dhi_orig = f['dhi'][:, sites]
        fill_orig = f['fill_flag'][:, sites]

    return ghi_orig, dni_orig, dhi_orig, fill_orig


def get_source_data(test_file=TEST_FILE, sites=list(range(10))):
    """Retrieve the variables required to run all-sky for a given set of sites.
    """
    out = {}
    var_list = ('surface_pressure', 'surface_albedo', 'ssa', 'aod', 'alpha',
                'ozone', 'solar_zenith_angle', 'total_precipitable_water',
                'asymmetry')

    with h5py.File(test_file, 'r') as f:

        # get unscaled source variables
        out['time_index'] = pd.to_datetime(f['time_index'][...].astype(str))

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


def run_all_sky(sites=list(range(10)), debug=False):
    """Run the all-sky processing code over the specified site list."""

    source_vars = get_source_data(sites=sites)

    # run all_sky processing
    all_sky_out = all_sky(**source_vars, debug=debug)
    return all_sky_out


def make_df(site):
    """Get dataframes containing single-site timeseries data for checking."""

    d = get_source_data(sites=site)

    dhi, dni, ghi, cs_dhi, cs_dni, cs_ghi, fill_flag, csr = run_all_sky(
        sites=site, debug=True)

    ghi_orig, dni_orig, dhi_orig, fill_orig = get_benchmark_data(sites=site)

    df_dhi = pd.DataFrame({'ti': d['ti'],
                           'sza': d['sza'].flatten(),
                           'cloud_type': d['cloud_type'].flatten(),
                           'fill_flag': fill_flag.flatten(),
                           'fill_flag_orig': fill_orig.flatten(),
                           'cld_opd': d['cld_opd_dcomp'].flatten(),
                           'dhi': dhi.flatten(),
                           'dhi_orig': dhi_orig.flatten(),
                           'cs_dhi': cs_dhi.flatten(),
                           })
    df_dni = pd.DataFrame({'ti': d['ti'],
                           'sza': d['sza'].flatten(),
                           'cloud_type': d['cloud_type'].flatten(),
                           'fill_flag': fill_flag.flatten(),
                           'fill_flag_orig': fill_orig.flatten(),
                           'cld_opd': d['cld_opd_dcomp'].flatten(),
                           'dni': dni.flatten(),
                           'dni_orig': dni_orig.flatten(),
                           'cs_dni': cs_dni.flatten(),
                           })
    df_ghi = pd.DataFrame({'ti': d['ti'],
                           'sza': d['sza'].flatten(),
                           'cloud_type': d['cloud_type'].flatten(),
                           'fill_flag': fill_flag.flatten(),
                           'fill_flag_orig': fill_orig.flatten(),
                           'cld_opd': d['cld_opd_dcomp'].flatten(),
                           'ghi': ghi.flatten(),
                           'ghi_orig': ghi_orig.flatten(),
                           'cs_ghi': cs_ghi.flatten(),
                           })

    return df_dhi, df_dni, df_ghi


def plot_benchmark(sites, y_range=None):
    """Make plots benchmarking allsky irradiance against a baseline set of
    irradiances.
    """
    dhi, dni, ghi = run_all_sky(sites=sites, debug=False)

    ghi_orig, dni_orig, dhi_orig, fill_orig = get_benchmark_data(sites=sites)

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


def test_all_sky(sites=list(range(10)), bad_threshold=0.005):
    """Run a numerical test of all_sky irradiance vs. benchmark NSRDB data."""
    dhi, dni, ghi = run_all_sky(sites=sites, debug=False)

    ghi_orig, dni_orig, dhi_orig, fill_orig = get_benchmark_data(sites=sites)

    max_perc_bad = 0

    for i, site in enumerate(sites):
        hist, bin_edges = np.histogram(np.abs(ghi[:, i] - ghi_orig[:, i]),
                                       bins=100, range=(0.0, 1000.0))

        n_bad = np.sum(hist[2:])
        frac_bad = n_bad / ghi.shape[0]
        max_perc_bad = np.max((max_perc_bad, 100 * frac_bad))

        msg = ('{0:.4f}% of the values do not match the baseline '
               'irradiance (threshold is {1:.4f}%) for site {2}.'
               .format(100 * frac_bad, 100 * bad_threshold, site))
        assert frac_bad < bad_threshold, msg

    print('Maximum of {0:.4f}% bad timesteps. Threshold was {1:.4f}%.'
          .format(max_perc_bad, 100 * bad_threshold))


def iter_speed_compare(sites=list(range(10))):
    """Run a speed test comparing broadcasted all-sky to iterated all-sky."""

    t0 = time.time()
    run_all_sky(sites=sites, debug=False)
    t_broad = time.time() - t0

    t0 = time.time()
    for site in sites:
        run_all_sky(sites=[site], debug=False)
    t_iter = time.time() - t0

    print('Running {0} sites through all sky took {1:.2f} seconds '
          'broadcasted, {2:.2f} seconds iterated.'
          .format(len(sites), t_broad, t_iter))


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
    execute_pytest()
