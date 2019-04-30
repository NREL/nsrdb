'''
@author: Anthony Lopez
January 27, 2015

Adapted from Chris Gueymard's REST2 model
(Proc. ASES Conf., 2004; Solar Energy, 2008).
Details from the original Fortran Code:

c      Program REST2_v9.0c [v9 Compact]
C
c     Gueymard's REST2 model (Proc. ASES Conf., 2004; Solar Energy, 2008).
c==>   Compact version of Latest version (v9) used here (2014).
c     Version 9.0: Corrected calculation of aerosol transmittance, now
c                 with extended AOD range; revsed solar constant;
c                 improved and streamlined diffuse algorithms, now using
c                 asymmetry parameter as input.
c     Version 9.0c: Compact algorithm; features:
c           - assumes Z is known (to avoid solar position calculations)
c           - does not calculate PAR or illuminances
c           - does not print intermediate results
c
c NOTE 1: The NO2 vertical pathlength is defaulted here to fit unpolluted
c      conditions and accelerate calculations in this compact version.
c      For better results over polluted areas, use the regular REST2 version!
c
c NOTE 2: The input sanity check on lines 112-120 of the code may be
c         superfluous if the inputs have been tested in a previous step.
c
c++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import numpy as np
import collections
from warnings import warn
import concurrent.futures as cf
import gc

import nsrdb.all_sky.utilities as ut
from nsrdb.all_sky import SOLAR_CONSTANT
from nsrdb.all_sky import SZA_LIM


def op_mass(p, am, cosz, z):
    """Calculate optical masses.

    Parameters
    ----------
    p : np.ndarray
        Pressure in mbar.
    am : np.ndarray
        Optical mass parameter from SMARTS Model.
    cosz : np.ndarray
        Cosine of the solar zenith angle.
    z : np.ndarray
        Solar zenith angle in degrees.
    """

    masr = am * p / 1013.25
    masw = np.maximum(1.0, 1.0 / (cosz + 0.10648 * (z ** 0.11423) /
                                  (93.781 - z) ** 1.9203))
    maso = np.maximum(1.0, 1.0 / (cosz + 1.0651 * (z ** 0.6379) /
                                  (101.8 - z) ** 2.2694))
    return masr, masw, maso


def trans_1(p, am, cosz, z, ozone, w):
    """Calculate band 1 transmittances.

    Parameters
    ----------
    p : np.ndarray
        Pressure in mbar.
    am : np.ndarray
        Optical mass parameter from SMARTS Model.
    cosz : np.ndarray
        Cosine of the solar zenith angle.
    z : np.ndarray
        Solar zenith angle in degrees.
    ozone : np.ndarray
        reduced ozone vertical pathlength (atm-cm)
        [Note: 1 atm-cm = 1000 DU]
    w : np.ndarray
        Total precip. water (cm).

    Returns
    -------
    transr1 : np.ndarray
        Transmittance for rayleigh scattering.
    transg1 : np.ndarray
        Transmittance for uniformly mixed gases.
    trano1 : np.ndarray
        Transmittance for ozone absorption.
    trann1 : np.ndarray
        Transmittance for nitrogen dioxide absorption.
    tranw1 : np.ndarray
        Transmittance for water absorption
    """

    # get the optical masses
    masr, masw, maso = op_mass(p, am, cosz, z)

    transr1 = ((1.0 + 1.8169 * masr - 0.033454 * masr ** 2) /
               (1.0 + 2.063 * masr + 0.31978 * masr ** 2))
    transg1 = ((1.0 + 0.95885 * masr + 0.012871 * masr ** 2) /
               (1.0 + 0.96321 * masr + 0.015455 * masr ** 2))

    a1 = (ozone * (10.979 - 8.5421 * ozone) /
          (1.0 + 2.0115 * ozone + 40.189 * ozone ** 2))
    a2 = (ozone * (-0.027589 - 0.005138 * ozone) /
          (1.0 - 2.4857 * ozone + 13.942 * ozone ** 2))
    a3 = (ozone * (10.995 - 5.5001 * ozone) /
          (1.0 + 1.6784 * ozone + 42.406 * ozone ** 2))
    trano1 = (1.0 + a1 * maso + a2 * maso ** 2) / (1.0 + a3 * maso)

    trann1 = ((1.0 + 0.18307 * masw - 0.00024 * masw ** 2) /
              (1.0 + 0.18713 * masw))

    c1 = w * (0.065445 + 0.00029901 * w) / (1.0 + 1.2728 * w)
    c2 = w * (0.065687 + 0.0013218 * w) / (1.0 + 1.2008 * w)
    tranw1 = (1.0 + c1 * masw) / (1.0 + c2 * masw)

    return transr1, transg1, trano1, trann1, tranw1


def trans_2(p, am, cosz, z, w):
    """Calculate band 2 transmittances.

    Parameters
    ----------
    p : np.ndarray
        Pressure in mbar.
    am : np.ndarray
        Optical mass parameter from SMARTS Model.
    cosz : np.ndarray
        Cosine of the solar zenith angle.
    z : np.ndarray
        Solar zenith angle in degrees.
    w : np.ndarray
        Total precip. water (cm).

    Returns
    -------
    transr2 : np.ndarray
        Transmittance for rayleigh scattering.
    transg2 : np.ndarray
        Transmittance for uniformly mixed gases.
    tranw2 : np.ndarray
        Transmittance for water absorption
    """

    # get the optical masses
    masr, masw, _ = op_mass(p, am, cosz, z)

    transr2 = (1.0 - 0.010394 * masr) / (1.0 - 0.00011042 * masr ** 2)
    trang2 = ((1.0 + 0.27284 * masr - 0.00063699 * masr ** 2) /
              (1.0 + 0.30306 * masr))

    c1 = w * ((19.566 - 1.6506 * w + 1.0672 * w ** 2) /
              (1.0 + 5.4248 * w + 1.6005 * w ** 2))
    c2 = w * ((0.50158 - 0.14732 * w + 0.047584 * w ** 2) /
              (1.0 + 1.1811 * w + 1.0699 * w ** 2))
    c3 = w * ((21.286 - 0.39232 * w + 1.2692 * w ** 2) /
              (1.0 + 4.8318 * w + 1.412 * w ** 2))
    c4 = w * ((0.70992 - 0.23155 * w + 0.096514 * w ** 2) /
              (1.0 + 0.44907 * w + 0.75425 * w ** 2))
    tranw2 = ((1.0 + c1 * masw + c2 * masw ** 2) /
              (1.0 + c3 * masw + c4 * masw ** 2))

    return transr2, trang2, tranw2


def band_1(alpha, masa, beta):
    """Get band (wavelength) 1.

    Parameters
    ----------
    alpha : np.ndarray
        Angstrom wavelength exponent.
    masa : np.ndarray
        Optical mass parameter from SMARTS Model.
    beta : np.ndarray
        Angstrom turbidity coeff.
    """

    wvlmin1 = 0.5158 - 0.008334 * alpha
    wvlmax1 = np.maximum(0.61, (0.6 + 0.95155 * alpha) /
                         (1.0 + 1.3095 * alpha))

    if np.any(alpha < 0):
        i = np.where(alpha < 0)
        wvlmin1[i] = 0.51
        wvlmax1[i] = 0.61

    # Coefficients for Band 1

    cc0 = (0.50947 - 0.012555 * alpha + 0.0026455 * alpha ** 2 +
           0.0044092 * alpha ** 3 - 0.0022439 * alpha ** 4 +
           0.0003123 * alpha ** 5)
    cc1 = (0.062836 + 0.049194 * alpha + 0.013976 * alpha ** 2 -
           0.0114290 * alpha ** 3 + 0.0053573 * alpha ** 4 +
           0.0026402 * alpha ** 5)
    cc2 = (0.096418 + 0.072221 * alpha + 0.015505 * alpha ** 2 -
           0.0216490 * alpha ** 3 + 0.0119010 * alpha ** 4 +
           0.0033763 * alpha ** 5)
    dd0 = 0.5
    dd1 = (0.065180 - 0.039075 * alpha + 0.11648 * alpha ** 2 +
           0.048987 * alpha ** 3 - 0.026766 * alpha ** 4 -
           0.12573 * alpha ** 5 + 0.092131 * alpha ** 6)
    dd2 = (0.099191 - 0.083962 * alpha + 0.20562 * alpha ** 2 +
           0.057377 * alpha ** 3 - 0.049548 * alpha ** 4 -
           0.17782 * alpha ** 5 + 0.13647 * alpha ** 6)

    y1 = masa * beta ** (0.3333 * alpha)
    z1 = masa * beta ** 0.5
    wvle1 = (cc0 + cc1 * y1) / (1.0 + cc2 * y1)

    if np.any(masa * beta > 10):
        i = np.where(masa * beta > 10)
        wvle1[i] = (dd0 + dd1[i] * z1[i]) / (1.0 + dd2[i] * z1[i])
    wvle1 = np.minimum(wvle1, wvlmax1)
    wvle1 = np.maximum(wvle1, wvlmin1)

    return wvle1


def band_2(alpha, masa, beta):
    """Get band (wavelength) 2.

    Parameters
    ----------
    alpha : np.ndarray
        Angstrom wavelength exponent.
    masa : np.ndarray
        Optical mass parameter from SMARTS Model.
    beta : np.ndarray
        Angstrom turbidity coeff.
    """

    wvlmin2 = 1.0 - 0.02 * alpha
    wvlmax2 = 1.3 + 1.5317 * alpha - 0.55289 * alpha ** 2

    if np.any(alpha < 0):
        i = np.where(alpha < 0)
        wvlmin2[i] = 1.0 + 0.4 * alpha[i]
        wvlmax2[i] = 1.3 + 0.25 * alpha[i]

    # Coefficients for Band 2

    aa0 = (1.0677 - 0.05432 * alpha + 0.014351 * alpha ** 2 -
           0.0097063 * alpha ** 3 + 0.0023655 * alpha ** 4)
    aa1 = (-0.20914 - 0.27218 * alpha + 0.83552 * alpha ** 2 -
           0.85437 * alpha ** 3 + 0.49305 * alpha ** 4 -
           0.14965 * alpha ** 5 + 0.018964 * alpha ** 6)
    aa2 = (0.0010588 + 0.039597 * alpha + 0.006733 * alpha ** 2 +
           0.070698 * alpha ** 3 - 0.11284 * alpha ** 4 +
           0.055096 * alpha ** 5 - 0.0086265 * alpha ** 6)
    aa3 = (-0.19432 - 0.29366 * alpha + 0.83474 * alpha ** 2 -
           0.78019 * alpha ** 3 + 0.37382 * alpha ** 4 -
           0.089069 * alpha ** 5 + 0.0091113 * alpha ** 6)

    x2 = np.log(1.0 + masa * beta)
    x22 = x2 * x2
    wvle2 = (aa0 + aa1 * x2 + aa2 * x22) / (1.0 + aa3 * x2)

    wvle2 = np.minimum(wvle2, wvlmax2)
    wvle2 = np.maximum(wvle2, wvlmin2)

    return wvle2


def aer_scat_trans(wvle, piar, alpha, beta, masa):
    """Calculate the aerosol scattering transmittances for a single band.

    Parameters
    ----------
    wvle : np.ndarray
        Wavelength for the single band.
    piar : np.ndarray
        Adjusted ssa parameter.
    alpha : np.ndarray
        Angstrom wavelength exponent.
    beta : np.ndarray
        Angstrom turbidity coeff.
    masa : np.ndarray
        Optical mass parameter from SMARTS Model.

    Returns
    -------
    tranas : np.ndarray
        T_as from eq 7b in ref [1]: Aerosol scattering transmittance.
    tranaa : np.ndarray
        Transmittance for spectral aerosol extinction from ABSORPTION
    tauas : np.ndarray
        Spectral aerosol optical depth along vertical atmospheric column
        (taua, eq 6 in ref [1]) times the single-scattering albedo (ssa).
    """

    # AEROSOL TRANSMITTANCES FOR THE 2 BANDS
    taua = beta / wvle ** alpha  # [1] eq. (6)
    tauas = piar * taua
    amsbet = masa * taua
    tranas = np.exp(-masa * tauas)  # [1] eq. (7b)
    tranaa = np.exp(-amsbet * (1.0 - piar))

    return tranas, tranaa, tauas


def calc_eabs(tabs, f, radius):
    """Calculate the dni for a given band without aerosol scattering.

    This method implements part of eq 3 from ref [1].

    Parameters
    ----------
    tabs : np.ndarray
        Total Absorption transmittances (product of T variables in eq 3
        from ref [1] without scattering terms).
    f : float
        Constant value, originally 0.47244 and 0.51951 for bands 1 and 2.
    radius : np.ndarray
        Sun-earth radius vector, varies between 1.017 in July and
        0.983 in January.

    Returns
    -------
    eabs : np.ndarray
        Direct normal irradiance for a given band in W/m2 without scattering.
    """

    # etdirn : Extraterrestrial direct normal irradiance at time t [W/m2]
    # etdirn = SOLAR_CONSTANT / (radius ** 2)
    eabs = tabs * f * SOLAR_CONSTANT / (radius ** 2)
    return eabs


def calc_edni(transr, tranas, eabs):
    """Calculate the dni for a given band with aerosol scattering.

    This method implements eq 3 from ref [1].

    Parameters
    ----------
    transr : np.ndarray
        Transmittance for rayleigh scattering.
    tranas : np.ndarray
        T_as from eq 7b in ref [1]: Aerosol scattering transmittance.
    eabs : np.ndarray
        Direct normal irradiance for a given band in W/m2 without scattering.

    Returns
    -------
    edni : np.ndarray
        Direct normal irradiance for a given band in W/m2 with aerosol
        scattering.
    """
    edni = eabs * transr * tranas
    return edni


def layer_props(transr, tauas, tranas, tabs, f, radius, cosz, albedo,
                am, fm, g0):
    """Get the layer properties for a single wavelength band"""

    # 1. Top layer properties (absorbing only)
    eabs = calc_eabs(tabs, f, radius)
    edni = calc_edni(transr, tranas, eabs)

    # 2. Bottom layer properties (scattering only)
    taur = -np.log(transr) / am
    taut = taur + tauas

    frwd1 = ((0.5 + 1.8823 * cosz) / (1.0 + 1.7971 * cosz) * tauas +
             0.5 * taur)
    amt = am * taut
    fn = (1.0 - 10.921 * amt - 11.741 * amt * amt) / (1.0 + 35.006 * amt)

    rsky = taut * (0.51754 + 0.15884 * taut) / (1.0 + 2.77 * taut)

    g1 = g0 * tauas / taut
    fg = (-0.5 + 10.497 * g1 - 11.735 * g1 ** 2) / (1.0 + 401.0 * g1 ** 2)
    edif = frwd1 * np.exp(fn + fg + fm) * eabs
    edift = edif + albedo * rsky * (edni * cosz + edif) / (1.0 - albedo * rsky)

    return edni, edif, edift, rsky


def rest2_parallel(p, albedo, ssa, g, z, radius, alpha, beta, ozone, w,
                   sza_lim=SZA_LIM, n_workers=16):
    """REST2 Clear Sky parallel execution method."""

    futures = []
    x_p = np.array_split(p, n_workers)
    x_albedo = np.array_split(albedo, n_workers)
    x_ssa = np.array_split(ssa, n_workers)
    x_g = np.array_split(g, n_workers)
    x_z = np.array_split(z, n_workers)
    x_radius = np.array_split(radius, n_workers)
    x_alpha = np.array_split(alpha, n_workers)
    x_beta = np.array_split(beta, n_workers)
    x_ozone = np.array_split(ozone, n_workers)
    x_w = np.array_split(w, n_workers)

    with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(rest2, x_p[i], x_albedo[i], x_ssa[i],
                                   x_g[i], x_z[i], x_radius[i], x_alpha[i],
                                   x_beta[i], x_ozone[i], x_w[i],
                                   sza_lim=sza_lim)
                   for i in range(n_workers)]

        futures = [future.result() for future in futures]

    var_list = ('ghi', 'dni', 'dhi', 'Tddclr', 'Tduclr', 'Ruuclr')
    rest_data = collections.namedtuple('rest_data', var_list)

    for i, future in enumerate(futures):
        for var in var_list:
            setattr(rest_data, var,
                    np.concatenate((getattr(rest_data, var),
                                    getattr(future, var)), axis=1))
    return rest_data


def rest2(p, albedo, ssa, g, z, radius, alpha, beta, ozone, w,
          sza_lim=SZA_LIM):
    """REST2 Clear Sky Model.

    Literature
    ----------
    [1] Christian A. Gueymard, "REST2: High-performance solar radiation model
        for cloudless-sky irradiance, illuminance, and photosynthetically
        active radiation – Validation with a benchmark dataset", Solar Energy,
        Volume 82, Issue 3, 2008, Pages 272-285, ISSN 0038-092X,
        https://doi.org/10.1016/j.solener.2007.04.008.
        (http://www.sciencedirect.com/science/article/pii/S0038092X07000990)

    Variables
    ---------
    eabs1 / eabs2
        Band (1 or 2) direct normal irradiance. (eq 3 from ref [1])
    tranas
        Transmittance for spectral aerosol extinction from SCATTERING
        (eq 7b in ref [1]).
    tranaa
        Transmittance for spectral aerosol extinction from ABSORPTION
    transg
        Transmittance for uniformly mixed gases absorption
    trann
        Transmittance for Nitrogen dioxide absorption
    tranw
        Transmittance for water absorption
    trano
        Transmittance for Ozone absorption
    taua
        Spectral aerosol optical depth along vertical atmospheric column.
        (eq 6 in ref [1])
    tauas
        taua times the single-scattering albedo (ssa)
    masa
        Relative slant pathlength or "air mass".

    Parameters
    ----------
    p : np.ndarray
        Surface pressure (mbar). If the range is very high, Pa are implied and
        the array is scaled to mbar with a warning.
    albedo : np.ndarray
        Ground albedo.
    ssa : np.ndarray
        aerosol single-scattering albedo at a representative wavelength of
        about 700 nm. Use -9.99 or any other NEGATIVE value if unknown
        (will default to 0.92 if a negative value is input)
    g : np.ndarray
        Aerosol asymmetry parameter. Since it tends to vary with wavelength
        and alpha, use a representative value for a wavelength of about
        700 nm and alpha about 1. Will default to 0.7 if a NEGATIVE value is
        input (when exact value is unknown).
    z : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    radius : np.ndarray
        Sun-earth radius vector, varies between 1.017 in July and
        0.983 in January.
    alpha : np.ndarray
        Angstrom wavelength exponent, ideally obtained by linear
        regression of all available spectral AODs between 380 and 1020 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.5], and corrected if necessary.
    beta : np.ndarray
        Angstrom turbidity coeff., i.e. AOD at 1000 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.2], and corrected if necessary.
        Can be calculated as beta=aod * np.power(0.55, alpha)
    ozone : np.ndarray
        reduced ozone vertical pathlength (atm-cm)
        [Note: 1 atm-cm = 1000 DU]
    w : np.ndarray
        Total precip. water (cm).
    sza_lim : float | int
        Upper limit for solar zenith angle in degrees. SZA values greater than
        this will be truncated at this value.

    Returns
    -------
    rest_data : collections.namedtuple
        Named tuple with clear sky irradiance data. Attributes:
            ghi : global horizontal irradiance (w/m2)
            dni : direct normal irradiance (w/m2)
            dhi : diffuse horizontal irradiance (w/m2)
            Tddclr : Transmittance of the clear-sky atmosphere for
                direct incident and direct outgoing fluxes (dd).
                Tddclr = dni / etdirn
            Tduclr : Transmittance of the clear-sky atmosphere for
                direct incident and diffuse outgoing fluxes (du).
                Tduclr = dhi / (etdirn * cosz)
            Ruuclr : Aerosol reflectance for diffuse incident and diffuse
                outgoing fluxes (uu).
    """

    # enforce ceiling on solar zenith angle
    z = np.where(z > sza_lim, sza_lim, z)

    if np.max(p) > 10000:
        # greater than 10 atmospheres in Pa, must be mbar
        warn('Surface pressure greater than 10,000. Assuming this was input '
             'as Pascal instead of mbar. Scaling by 0.01 to convert to mbar')
        p *= 0.01

    # Constants
    degrad = 0.017453293
    f1 = 0.47244
    f2 = 0.51951

    # Input Verification
    alpha = np.maximum(0.0, np.minimum(alpha, 2.501))
    beta = np.maximum(0.001, np.minimum(beta, 2.2))
    albedo = np.maximum(0.05, np.minimum(albedo, 9))

    cosz = np.cos(z * degrad)

    g0 = g + 0.066 * (1.0 - alpha)
    piar1 = np.maximum(0.85, np.minimum(0.98, ssa + 0.03))
    piar2 = np.minimum(0.95, np.maximum(0.85, ssa - 0.02))

    # Vectorized operation
    piar1 = np.where(ssa <= 0, 0.95, piar1)
    piar2 = np.where(ssa <= 0, 0.90, piar2)
    ssa = np.where(ssa <= 0, 0.92, ssa)

    # Optical Masses from SMARTS Model
    am = np.maximum(1.0, 1.0 / (cosz + 0.48353 * (z ** 0.095846) /
                                (96.741 - z) ** 1.754))
    masa = np.maximum(1.0, 1.0 / (cosz + 0.16851 * (z ** 0.18198) /
                                  (95.318 - z) ** 1.9542))

    # calculate the transmittances for both bands
    transr1, transg1, trano1, trann1, tranw1 = trans_1(p, am, cosz, z,
                                                       ozone, w)
    transr2, trang2, tranw2 = trans_2(p, am, cosz, z, w)

    # New aerosol functions in v9

    ambeta = masa * beta

    # get the two band wavelengths
    wvle1 = band_1(alpha, masa, beta)
    wvle2 = band_2(alpha, masa, beta)

    # AOD correction
    aodcor = ambeta * (0.015981 + 0.183 * ambeta) / (1.0 + 1.4142 * ambeta)

    # get the aerosol scattering and absorption transmittances
    tranas1, tranaa1, tauas1 = aer_scat_trans(wvle1, piar1, alpha, beta, masa)
    tranas2, tranaa2, tauas2 = aer_scat_trans(wvle2, piar2, alpha, beta, masa)

    # Total Absorption transmittances
    tabs1 = transg1 * trann1 * tranw1 * trano1 * tranaa1
    tabs2 = trang2 * tranw2 * tranaa2

    # DIFFUSE ALGORITHM
    fm = 0.15244 * (am - 1.0) / (1.0 + 2.2413 * am)

    # Get BAND1 layer properties : UV-VIS PART OF THE SPECTRUM (0.3-0.7 um)
    edni1, edif1, edift1, rsky1 = layer_props(transr1, tauas1, tranas1,
                                              tabs1, f1, radius, cosz,
                                              albedo, am, fm, g0)
    # Get BAND2 layer properties : INFRA-RED PART OF THE SPECTRUM (0.7-4 um)
    edni2, edif2, edift2, rsky2 = layer_props(transr2, tauas2, tranas2,
                                              tabs2, f2, radius, cosz,
                                              albedo, am, fm, g0)
    # Broadband results
    edif = edift1 + edift2
    edirn = (edni1 + edni2) * np.exp(aodcor)
    eglob = edif + edirn * cosz

    # Extra Return values for FastModel
    Tddclr = edirn / (SOLAR_CONSTANT / (radius ** 2))
    Tduclr = edif / ((SOLAR_CONSTANT / (radius ** 2)) * cosz)
    Ruuclr = (((edni1 * cosz + edif1) * rsky1 +
               (edni2 * cosz + edif2) * rsky2) /
              ((edni1 * cosz + edif1) + (edni2 * cosz + edif2)))

    ut.check_range(Tddclr, 'Tddclr')
    ut.check_range(Tduclr, 'Tduclr')
    ut.check_range(Ruuclr, 'Ruuclr')

    rest_data = collections.namedtuple('rest_data', ['ghi', 'dni', 'dhi',
                                                     'Tddclr', 'Tduclr',
                                                     'Ruuclr'])
    rest_data.ghi = eglob
    rest_data.dni = edirn
    rest_data.dhi = edif
    rest_data.Tddclr = Tddclr
    rest_data.Tduclr = Tduclr
    rest_data.Ruuclr = Ruuclr

    return rest_data


def rest2_tddclr(p, albedo, ssa, z, radius, alpha, beta, ozone, w,
                 sza_lim=SZA_LIM):
    """REST2 Clear Sky Model for only calculating Tddclr for FARMS input.

    Literature
    ----------
    [1] Christian A. Gueymard, "REST2: High-performance solar radiation model
        for cloudless-sky irradiance, illuminance, and photosynthetically
        active radiation – Validation with a benchmark dataset", Solar Energy,
        Volume 82, Issue 3, 2008, Pages 272-285, ISSN 0038-092X,
        https://doi.org/10.1016/j.solener.2007.04.008.
        (http://www.sciencedirect.com/science/article/pii/S0038092X07000990)

    Variables
    ---------
    eabs1 / eabs2
        Band (1 or 2) direct normal irradiance. (eq 3 from ref [1])
    tranas
        Transmittance for spectral aerosol extinction from SCATTERING
        (eq 7b in ref [1]).
    tranaa
        Transmittance for spectral aerosol extinction from ABSORPTION
    transg
        Transmittance for uniformly mixed gases absorption
    trann
        Transmittance for Nitrogen dioxide absorption
    tranw
        Transmittance for water absorption
    trano
        Transmittance for Ozone absorption
    taua
        Spectral aerosol optical depth along vertical atmospheric column.
        (eq 6 in ref [1])
    tauas
        taua times the single-scattering albedo (ssa)
    masa
        Relative slant pathlength or "air mass".

    Parameters
    ----------
    p : np.ndarray
        Surface pressure (mbar). If the range is very high, Pa are implied and
        the array is scaled to mbar with a warning.
    albedo : np.ndarray
        Ground albedo.
    ssa : np.ndarray
        aerosol single-scattering albedo at a representative wavelength of
        about 700 nm. Use -9.99 or any other NEGATIVE value if unknown
        (will default to 0.92 if a negative value is input)
    z : np.ndarray
        Solar zenith angle (degrees). Must represent the average value over the
        integration period (e.g. hourly) under scrutiny.
    radius : np.ndarray
        Sun-earth radius vector, varies between 1.017 in July and
        0.983 in January.
    alpha : np.ndarray
        Angstrom wavelength exponent, ideally obtained by linear
        regression of all available spectral AODs between 380 and 1020 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.5], and corrected if necessary.
    beta : np.ndarray
        Angstrom turbidity coeff., i.e. AOD at 1000 nm.
        Input value will be tested for compliance with the mandatory
        interval [0, 2.2], and corrected if necessary.
        Can be calculated as beta=aod * np.power(0.55, alpha)
    ozone : np.ndarray
        reduced ozone vertical pathlength (atm-cm)
        [Note: 1 atm-cm = 1000 DU]
    w : np.ndarray
        Total precip. water (cm).
    sza_lim : float | int
        Upper limit for solar zenith angle in degrees. SZA values greater than
        this will be truncated at this value.

    Returns
    -------
    rest_data : collections.namedtuple
        Named tuple with clear sky irradiance data. Attributes:
            ghi : global horizontal irradiance (w/m2)
            dni : direct normal irradiance (w/m2)
            dhi : diffuse horizontal irradiance (w/m2)
            Tddclr : Transmittance of the clear-sky atmosphere for
                direct incident and direct outgoing fluxes (dd).
                Tddclr = dni / etdirn
            Tduclr : Transmittance of the clear-sky atmosphere for
                direct incident and diffuse outgoing fluxes (du).
                Tduclr = dhi / (etdirn * cosz)
            Ruuclr : Aerosol reflectance for diffuse incident and diffuse
                outgoing fluxes (uu).
    """

    # enforce ceiling on solar zenith angle
    z = np.where(z > sza_lim, sza_lim, z)

    if np.max(p) > 10000:
        # greater than 10 atmospheres in Pa, must be mbar
        warn('Surface pressure greater than 10,000. Assuming this was input '
             'as Pascal instead of mbar. Scaling by 0.01 to convert to mbar')
        p *= 0.01

    # Constants
    degrad = 0.017453293
    f1 = 0.47244
    f2 = 0.51951

    # Input Verification
    alpha = np.maximum(0.0, np.minimum(alpha, 2.501))
    beta = np.maximum(0.001, np.minimum(beta, 2.2))
    albedo = np.maximum(0.05, np.minimum(albedo, 9))

    cosz = np.cos(z * degrad)

    piar1 = np.maximum(0.85, np.minimum(0.98, ssa + 0.03))
    piar2 = np.minimum(0.95, np.maximum(0.85, ssa - 0.02))

    # Vectorized operation
    piar1 = np.where(ssa <= 0, 0.95, piar1)
    piar2 = np.where(ssa <= 0, 0.90, piar2)
    ssa = np.where(ssa <= 0, 0.92, ssa)

    # Optical Masses from SMARTS Model
    am = np.maximum(1.0, 1.0 / (cosz + 0.48353 * (z ** 0.095846) /
                                (96.741 - z) ** 1.754))
    masa = np.maximum(1.0, 1.0 / (cosz + 0.16851 * (z ** 0.18198) /
                                  (95.318 - z) ** 1.9542))

    # calculate the transmittances for both bands
    transr1, transg1, trano1, trann1, tranw1 = trans_1(p, am, cosz, z,
                                                       ozone, w)
    transr2, trang2, tranw2 = trans_2(p, am, cosz, z, w)

    # clear some variables to free memory
    del p, am, cosz, z, w

    # New aerosol functions in v9
    ambeta = masa * beta

    # get the two band wavelengths
    wvle1 = band_1(alpha, masa, beta)
    wvle2 = band_2(alpha, masa, beta)

    # AOD correction
    aodcor = ambeta * (0.015981 + 0.183 * ambeta) / (1.0 + 1.4142 * ambeta)

    # get the aerosol scattering and absorption transmittances
    tranas1, tranaa1, _ = aer_scat_trans(wvle1, piar1, alpha, beta, masa)
    tranas2, tranaa2, _ = aer_scat_trans(wvle2, piar2, alpha, beta, masa)

    # clear some variables to free memory
    del alpha, beta, masa, piar1, piar2, wvle1, wvle2, ssa

    # Total Absorption transmittances
    tabs1 = transg1 * trann1 * tranw1 * trano1 * tranaa1
    tabs2 = trang2 * tranw2 * tranaa2

    # clear some variables to free memory
    del transg1, trann1, tranw1, trano1, tranaa1, trang2, tranw2, tranaa2

    # calculate edni
    edni1 = calc_edni(transr1, tranas1, calc_eabs(tabs1, f1, radius))
    edni2 = calc_edni(transr2, tranas2, calc_eabs(tabs2, f2, radius))

    # clear some variables to free memory
    del transr1, transr2, tranas1, tranas2, tabs1, tabs2

    # Broadband results
    edirn = (edni1 + edni2) * np.exp(aodcor)

    # Extra Return values for FastModel
    Tddclr = edirn / (SOLAR_CONSTANT / (radius ** 2))

    ut.check_range(Tddclr, 'Tddclr')

    return Tddclr


def rest2_tuuclr(p, albedo, ssa, radius, alpha, ozone, w, parallel=False,
                 diffuse_angles=(84.2608, 78.4630, 72.5424, 66.4218, 60.0000,
                                 53.1301, 45.5730, 36.8699, 25.8419, 0.00000)):
    """Calculate Tuuclr based on average values from several REST2 runs.

    Equation 5 from the following reference:
    [1] Yu Xie, Manajit Sengupta, Jimy Dudhia, "A Fast All-sky Radiation Model
        for Solar applications (FARMS): Algorithm and performance evaluation",
        Solar Energy, Volume 135, 2016, Pages 435-445, ISSN 0038-092X,
        https://doi.org/10.1016/j.solener.2016.06.003.
        (http://www.sciencedirect.com/science/article/pii/S0038092X16301827)

    Parameters
    ----------
    p : np.ndarray
        See rest2 doc string for description.
    albedo : np.ndarray
        See rest2 doc string for description.
    ssa : np.ndarray
        See rest2 doc string for description.
    radius : np.ndarray
        See rest2 doc string for description.
    alpha : np.ndarray
        See rest2 doc string for description.
    ozone : np.ndarray
        See rest2 doc string for description.
    w : np.ndarray
        See rest2 doc string for description.
    parallel : bool
        Flag to each diffuse angle on a seperate core using concurrent futures.
    diffuse_angles : list | tuple
        Set of solar zenith angles from 0 to near 90 used to calculate Tuuclr.

    Returns
    -------
    Tuuclr : np.ndarray
        Transmittance of the clear-sky atmosphere for diffuse incident and
        diffuse outgoing fluxes (uu).
    """

    Tddclr_list = []

    # ensure radius is of the correct shape
    if radius.shape != p.shape:
        radius = np.tile(radius, p.shape[1])

    if not parallel:
        # serial execution
        for angle in diffuse_angles:
            Tddclr_list.append(
                rest2_tddclr(p=p, albedo=albedo, ssa=ssa, z=angle,
                             radius=radius, alpha=alpha, beta=0,
                             ozone=ozone, w=w))
            gc.collect()
    else:
        # parallel execution
        n_workers = len(diffuse_angles)
        with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
            # submit futures for each angle
            futures = [executor.submit(rest2_tddclr, p, albedo, ssa, angle,
                                       radius, alpha, 0, ozone, w)
                       for angle in diffuse_angles]

            Tddclr_list = [future.result() for future in futures]
        gc.collect()

    scalar = 1 / (len(diffuse_angles))
    for i, angle in enumerate(diffuse_angles):
        Tddclr_list[i] = (Tddclr_list[i] * np.cos(np.radians(angle)) *
                          scalar)

    # Get the average for various angles
    Tuuclr = np.sum(np.array(Tddclr_list), axis=0) * 2.0

    del Tddclr_list
    gc.collect()

    return Tuuclr
