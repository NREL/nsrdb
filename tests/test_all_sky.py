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

    with h5py.File(test_file, 'r') as f:

        # get unscaled source variables
        ti = pd.to_datetime(f['time_index'][...].astype(str))
        p = (f['surface_pressure'][:, sites] /
             f['surface_pressure'].attrs['psm_scale_factor'])
        albedo = (f['surface_albedo'][:, sites] /
                  f['surface_albedo'].attrs['psm_scale_factor'])
        ssa = (f['ssa'][:, sites] / f['ssa'].attrs['psm_scale_factor'])
        aod = (f['aod'][:, sites] / f['aod'].attrs['psm_scale_factor'])
        alpha = (f['alpha'][:, sites] / f['alpha'].attrs['psm_scale_factor'])
        ozone = (f['ozone'][:, sites] / f['ozone'].attrs['psm_scale_factor'])
        sza = (f['solar_zenith_angle'][:, sites] /
               f['solar_zenith_angle'].attrs['psm_scale_factor'])
        w = (f['total_precipitable_water'][:, sites] /
             f['total_precipitable_water'].attrs['psm_scale_factor'])
        cloud_type = f['cloud_type'][:, sites]
        asym = (f['asymmetry'][:, sites] /
                f['asymmetry'].attrs['psm_scale_factor'])

        # different unscaling for cloud properties
        cld_reff_dcomp = (f['cld_reff_dcomp'][:, sites].astype(float) *
                          f['cld_reff_dcomp'].attrs['psm_scale_factor'] +
                          f['cld_reff_dcomp'].attrs['psm_add_offset'])
        cld_opd_dcomp = (f['cld_opd_dcomp'][:, sites].astype(float) *
                         f['cld_opd_dcomp'].attrs['psm_scale_factor'] +
                         f['cld_opd_dcomp'].attrs['psm_add_offset'])

    out = {'albedo': albedo,
           'alpha': alpha,
           'aod': aod,
           'asym': asym,
           'ozone': ozone,
           'p': p,
           'ssa': ssa,
           'sza': sza,
           'ti': ti,
           'w': w,
           'cloud_type': cloud_type,
           'cld_opd_dcomp': cld_opd_dcomp,
           'cld_reff_dcomp': cld_reff_dcomp}

    return out


def run_all_sky(test_file=TEST_FILE, sites=list(range(10)), debug=False):
    """Run the all-sky processing code over the specified site list."""

    source_vars = get_source_data(test_file=test_file, sites=sites)

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


def plot_benchmark(sites):
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

        # make center of x-axis the timestep of greatest error
        center = loc_max_diff[site]
        t0 = center - 50
        t1 = center + 50

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
    plot_benchmark(sites=list(range(10)))
