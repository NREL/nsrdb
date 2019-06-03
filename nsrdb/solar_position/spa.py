"""
NREL's solar position algorithm (SPA)
"""
import numpy as np
import pandas as pd


class SPA:
    """
    Solar position algorithm
    """
    HELIO_LONG_TABLE = None
    HELIO_LAT_TABLE = None
    HELIO_RADIUS_TABLE = None
    NUTATION_YTERM_ARRAY = None
    NUTATION_ABCD_ARRAY = None

    def __init__(self, time_index, lat_lon):
        """
        Parameters
        ----------
        time_index : ndarray | pandas.DatetimeIndex | str
            Datetime stamps of interest
        lat_lon : ndarray
            (latitude, longitude) for site(s) of interest
        """
        if not isinstance(time_index, pd.DatetimeIndex):
            if isinstance(time_index, str):
                time_index = [time_index]

            time_index = pd.to_datetime(time_index)

        self._time_index = time_index

        if not isinstance(lat_lon, np.ndarray):
            lat_lon = np.array(lat_lon)

        self._lat_lon = np.expand_dims(lat_lon, axis=0).T

    @property
    def time_index(self):
        """
        Datetime stamp(s) of interest

        Returns
        -------
        time_index : pandas.DatetimeIndex
        """
        return self._time_index

    @property
    def latitude(self):
        """
        Latitudes of site(s)

        Returns
        -------
        lat : ndarray
        """
        lat = self._lat_lon[0]
        return lat

    @property
    def longitude(self):
        """
        longitude of site(s)

        Returns
        -------
        lon : ndarray
        """
        lon = self._lat_lon[1]
        return lon

    @staticmethod
    def _parse_time(time_index, delta_t=67):
        """
        Convert UTC datetime index into:
        - Julian day
        - Julian ephemeris day
        - Julian century
        - Julian ephemeris century
        - Julian ephemeris millennium

        Parameters
        ----------
        time_index : pandas.DatetimeIndex
            Datetime stamps of interest
        delta_t : float
            Difference between terrestrial time and UT1

        Returns
        -------
        jd : ndarray
            Julian day
        jde : ndarray
            Julian ephemeris day
        jc : ndarray
            Julian century
        jce : ndarray
            Julian ephemeris century
        jme : ndarray
            Julian ephemeris millennium
        """
        jd = time_index.to_julian_date().values
        jde = jd + delta_t * 1 / 86400
        jc = (jd - 2451545) * 1 / 36525
        jce = (jde - 245154) * 1 / 36525
        jme = jce * 1 / 10

        return jd, jde, jc, jce, jme

    @staticmethod
    def _helicocentric_vector(arr, jme):
        """
        Perform heliocentric array to vector calculation:

        Parameters
        ----------
        arr : ndarray
            Array to compute vector from
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        out : ndarray
            heliocentric value for each input timestep
        """
        jme = np.expand_dims(jme, axis=0)
        out = np.sum(arr[:, :, [0]]
                     * np.cos(arr[:, :, [1]] + arr[:, :, [2]] * jme), axis=1)
        out = np.sum(out * np.power(jme.T, range(len(arr))).T, axis=0) / 10**8
        return out

    @staticmethod
    def heliocentric_longitude(helio_lon, jme):
        """
        Compute heliocentric Longitude

        Parameters
        ----------
        helio_lon : ndarray
            heliocentric longitude table
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        lon : ndarray
            Heliocentric longitude in degrees
        """
        lon = SPA._helicocentric_vector(helio_lon, jme)
        lon = np.rad2deg(lon) % 360
        return lon

    @staticmethod
    def heliocentric_latitude(helio_lat, jme):
        """
        Compute heliocentric latitude

        Parameters
        ----------
        helio_lat : ndarray
            heliocentric latitude table
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        b : ndarray
            Heliocentric longitude in degrees
        """
        b = SPA._helicocentric_vector(helio_lat, jme)
        b = np.rad2deg(b)
        return b

    @staticmethod
    def geocentric_longitude(heliocentric_longitude):
        """
        Compute geocentric longitude from heliocentric longitude

        Parameters
        ----------
        heliocentric_longitude : ndarray
            heliocentric longitude for each available timestep

        Returns
        -------
        theta : ndarray
            geocentric longitude for each available timestep
        """
        theta = heliocentric_longitude + 180.0
        return theta % 360

    @staticmethod
    def geocentric_latitude(heliocentric_latitude):
        """
        Compute geocentric latitude from heliocentric Latitude

        Parameters
        ----------
        heliocentric_latitude : ndarray
            heliocentric latitude for each available timestep

        Returns
        -------
        beta : ndarray
            geocentric latitude for each available timestep
        """
        beta = -1.0 * heliocentric_latitude
        return beta

    @staticmethod
    def mean_elongation(jce):
        """
        Compute mean elongation

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x0 : ndarray
            Mean elogation for all timesteps
        """
        x0 = (297.85036 + 445267.111480 * jce - 0.0019142 * jce**2 + jce**3
              / 189474)
        return x0

    @staticmethod
    def mean_anomaly_sun(jce):
        """
        Compute mean sun anomaly

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x1 : ndarray
            Mean sun anomaly for all timesteps
        """
        x1 = (357.52772 + 35999.050340 * jce - 0.0001603 * jce**2 - jce**3
              / 300000)
        return x1

    @staticmethod
    def mean_anomaly_moon(jce):
        """
        Compute mean moon anomaly

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x2 : ndarray
            Mean moon anomaly for all timesteps
        """
        x2 = (134.96298 + 477198.867398 * jce + 0.0086972 * jce**2 + jce**3
              / 56250)
        return x2

    @staticmethod
    def moon_argument_latitude(jce):
        """
        Compute moon latitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x3 : ndarray
            Moon latitude for all timesteps
        """
        x3 = (93.27191 + 483202.017538 * jce - 0.0036825 * jce**2 + jce**3
              / 327270)
        return x3

    @staticmethod
    def moon_ascending_longitude(jce):
        """
        Compute moon ascending longitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x4 : ndarray
            Moon ascending longitude for all timesteps
        """
        x4 = (125.04452 - 1934.136261 * jce + 0.0020708 * jce**2 + jce**3
              / 450000)
        return x4

    @staticmethod
    def nutation_coefficients(jce):
        """
        Compute the nutation coefficients:
        x0 = mean elongation
        x1 = mean sun anomaly
        x2 = mean moon anomaly
        x3 = moon latitude
        x4 = moon ascending longitude

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        x_arr : ndarray
            Array of nutation coeffiecients [x0, x1, x2, x3, x4]
        """
        x0 = SPA.mean_elongation(jce)
        x1 = SPA.mean_anomaly_sun(jce)
        x2 = SPA.mean_anomaly_moon(jce)
        x3 = SPA.moon_argument_latitude(jce)
        x4 = SPA.moon_ascending_longitude(jce)
        return np.array([x0, x1, x2, x3, x4])

    @staticmethod
    def mean_ecliptic_obliquity(jme):
        """
        Compute mean ecliptic obliquity

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium for all timesteps

        Returns
        -------
        e0 : ndarray
            Mean ecliptic obliquity for all timesteps
        """
        U = np.expand_dims(jme, axis=0).T / 10
        e0_coeff = np.array([84381.448, -4680.93, -1.55, 1999.25, -51.38,
                             -249.67, -39.05, 7.12, 27.87, 5.79, 2.45])

        e0 = np.sum(e0_coeff * np.power(U, range(11)), axis=1)
        return e0

    @staticmethod
    def true_ecliptic_obliquity(e0, delta_eps):
        """
        Compute true ecliptic obliquity

        Parameters
        ----------
        e0 : ndarray
            Mean elciptic obliquity
        delta_eps : ndarray
            Nutation obliquity

        Returns
        -------
        e : ndarray
            True ecliptic obliquity
        """
        e = e0 / 3600 + delta_eps
        return e

    @staticmethod
    def aberration_correction(r):
        """
        Compute aberration correction

        Parameters
        ----------
        r : ndarray
            Heliocentric or earth radius vector

        Returns
        -------
        delta_u : ndarray
            Aberation correction
        """
        delta_u = -20.4898 / (3600 * r)
        return delta_u

    @staticmethod
    def apparent_sun_longitude(beta, delta_psi, delta_u):
        """
        Compute apparent sun longitude

        Parameters
        ----------
        beta : ndarray
            Geocentric latitude
        delta_psi : ndarray
            Nutation longitude
        delta_u : ndarray
            Aberration correction

        Returns
        -------
        lamd : ndarray
            Apparent sun longitude
        """
        lamd = beta + delta_psi + delta_u
        return lamd

    @staticmethod
    def mean_sidereal_time(jd, jc):
        """
        Compute mean sidereal time

        Parameters
        ----------
        jd : ndarray
            Julian day for all timesteps
        jc : ndarray
            Julian century for all timesteps

        v0 : ndarray
            Mean sidereal time in degrees
        """
        v0 = (280.46061837 + 360.98564736629 * (jd - 2451545)
              + 0.000387933 * jc**2 - jc**3 / 38710000)
        return v0 % 360.0

    @staticmethod
    def apparent_sidereal_time(v0, delta_psi, e):
        """
        Compute apparent sidereal time

        Parameters
        ----------
        v0 : ndarray
            Mean sidreal time
        delta_psi : ndarray
            Nutation longitude
        e : ndarray
            True ecliptic obliquity

        Returns
        -------
        v : ndarray
            Apparent sidereal time degrees
        """
        v = v0 + delta_psi * np.cos(
            np.radians(e))
        return v

    @staticmethod
    def geocentric_sun_position(lamd, e, beta):
        """
        Compute geocentric sun position

        Parameters
        ----------
        lamd : ndarray
            Apparent sun longitude
        e : ndarray
            True ecliptic obliquity
        beta : ndarray
            Geocentric latitude

        Returns
        -------
        alpha : ndarray
            Geocentric sun right ascension in degrees
        delta : ndarray
            Geocentric sun delication in degrees
        """
        lamd = np.radians(lamd)
        e = np.radians(e)
        beta = np.radians(beta)
        num = (np.sin(lamd) * np.cos(e) - np.tan(beta) * np.sin(e))
        denom = np.cos(lamd)
        alpha = np.degrees(np.arctan2(num, denom)) % 360

        delta = (np.sin(beta) * np.cos(e) + np.cos(beta) * np.sin(e)
                 * np.sin(lamd))
        delta = np.degrees(np.arcsin(delta))

        return alpha, delta

    @staticmethod
    def local_hour_angle(v, obs_lon, alpha):
        """
        Compute local hour angle measured westward from south

        Parameters
        ----------
        v : ndarray
            Apparent sidreal time
        obs_lon : ndarray
            Observers longitudes
        alpha : ndarray
            Sun right ascension

        Returns
        -------
        H : ndarray
            Local hour angle in degrees from westward from south
        """
        H = v + obs_lon - alpha
        return H % 360

    @staticmethod
    def equatorial_horizontal_parallax(r):
        """
        Computes equatorial horizonatl parallax

        Parameters
        ----------
        r : ndarray
            heliocentric or earth radius vector

        Returns
        -------
        xi : ndarray
            Equatorial horizontal parallax
        """
        xi = 8.794 / (3600 * r)
        return xi

    @staticmethod
    def observer_xy(obs_lat, obs_elev):
        """
        Compute the observer x and y terms

        Parameters
        ----------
        obs_lat : ndarray
            Observers latitudes
        obs_elev : ndarray
            Observers elevations

        Returns
        -------
        obs_x : ndarray
            Observers x terms
        obs_y : ndarray
            Observers y terms
        """
        obs_lat = np.radians(obs_lat)
        u = np.arctan(0.99664719 * np.tan(obs_lat))
        obs_x = (np.cos(u) + obs_elev / 6378140 * np.cos(obs_lat))
        obs_y = (0.99664719 * np.sin(u) + obs_elev / 6378140 * np.sin(obs_lat))
        return obs_x, obs_y

    @staticmethod
    def parallax_sun_right_ascension(obs_x, xi, H, delta):
        """
        Computer sun right ascension parallax

        Parameters
        ----------
        obs_x : ndarray
            Observers x terms
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination

        Returns
        -------
        delta_alpha : ndarray
            Sun right ascension parallax
        """
        xi = np.radians(xi)
        H = np.radians(H)
        delta = np.radians(delta)
        num = (-obs_x * np.sin(xi) * np.sin(H))
        denom = (np.cos(delta) - obs_x * np.sin(xi) * np.cos(H))
        delta_alpha = np.degrees(np.arctan2(num, denom))
        return delta_alpha

    @staticmethod
    def topocentric_sun_declination(obs_x, obs_y, xi, H, delta, delta_alpha):
        """
        Compute topocentric sun position: right ascention and declination

        Parameters
        ----------
        obs_x : ndarray
            Observers x terms
        obs_y : ndarray
            Observers y terms
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination
        delta_alpha : ndarray
            Sun right ascensoin parallax

        Returns
        -------
        delta_prime : ndarray
            Topocentric sun declination angle in degrees
        """
        # Topocentric sun right ascension angle in degrees
        # alpha_prime = alpha + delta_alpha

        delta = np.radians(delta)
        xi = np.radians(xi)
        H = np.radians(H)
        delta_alpha = np.radians(delta_alpha)
        num = ((np.sin(delta) - obs_y * np.sin(xi)) * np.cos(delta_alpha))
        denom = (np.cos(delta) - obs_x * np.sin(xi) * np.cos(H))
        delta_prime = np.degrees(np.arctan2(num, denom))
        return delta_prime

    @staticmethod
    def topocentric_local_hour_angle(H, delta_alpha):
        """
        Compute topocentric local hour angle

        Parameters
        ----------
        H : ndarray
            Local hour angle
        delta_alpha : ndarray
            Sun right ascension parallax

        Returns
        -------
        H_prime : ndarray
            Topocentric local hour angle
        """
        H_prime = H - delta_alpha
        return H_prime

    @staticmethod
    def topocentric_solar_position(obs_lat, obs_elev, xi, H, delta):
        """
        Compute the topocentric sun position: elevation and azimuth
        - without atmospheric correction

        Parameters
        ----------
        obs_lat : ndarray
            Observers latitudes
        obs_elev : ndarray
            Observers elevations
        xi : ndarray
            Equatorial horizontal parallax
        H : ndarray
            Local hour angle
        delta : ndarray
            Geocentric sun declination

        Returns
        -------
        e0 : ndarray
            Topocentric elevation angle in degrees
        phi : ndarray
            Topocentric azimuth angle
        """
        obs_x, obs_y = SPA.observer_xy(obs_lat, obs_elev)
        delta_alpha = SPA.parallax_sun_right_ascension(obs_x, xi, H, delta)
        delta_prime = SPA.topocentric_sun_declination(obs_x, obs_y, xi, H,
                                                      delta, delta_alpha)
        H_prime = SPA.topocentric_local_hour_angle(H, delta_alpha)

        obs_lat = np.radians(obs_lat)
        delta_prime = np.radians(delta_prime)
        H_prime = np.radians(H_prime)
        e0 = (np.sin(obs_lat) * np.sin(delta_prime) + np.cos(obs_lat)
              * np.cos(delta_prime) * np.cos(H_prime))
        e0 = np.degrees(np.arcsin(e0))

        num = np.sin(H_prime)
        denom = (np.cos(H_prime) * np.sin(obs_lat) - np.tan(delta_prime)
                 * np.cos(obs_lat))
        gamma = np.degrees(np.arctan2(num, denom)) % 360
        phi = (gamma + 180) % 360
        return e0, phi

    @staticmethod
    def atmospheric_refraction_correction(pres, temp, e0,
                                          atmos_refract=0.5667):
        """
        Compute the atmospheric refraction correction value for all
        sites

        Parameters
        ----------
        pres : ndarray
            Pressure at all sites
        temp : ndarray
            Temperature at all sites
        e0 : ndarray
            Topocentric elevation angle
        atmos_refract : float
            Atmospheric refraction constant

        Returns
        -------
        delta_e : ndarray
            Atmospheric refraction correction
        """
        # switch sets delta_e when the sun is below the horizon
        switch = e0 >= -1.0 * (0.26667 + atmos_refract)
        angle = np.radians(e0 + 10.3 / (e0 + 5.11))
        delta_e = ((pres / 1010.0) * (283.0 / (273 + temp))
                   * 1.02 / (60 * np.tan(angle))) * switch
        return delta_e

    @staticmethod
    def apparent_elevation_angle(e0, delta_e):
        """
        The apparent topocentric elevation angle after refraction

        Parameters
        ----------
        e0 : ndarray
            Topocentric elevation angle
        delta_e : ndarray
            Atmospheric refraction correction

        Returns
        -------
        e : ndarray
            Apparent topocentric elevation angle after refraction
        """
        e = e0 + delta_e
        return e

    @staticmethod
    def topocentric_zenith_angle(e):
        """
        Topocentric zenith angle

        Parameters
        ----------
        e : ndarray
            Topocentric elevation angle

        Returns
        -------
        theta : ndarray
            Topocentric zenith angle
        """
        theta = 90 - e
        return theta

    @staticmethod
    def sun_mean_longitude(jme):
        """
        Compute mean sun longitude for all timesteps

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        M : ndarray
            Mean sun longitude
        """
        M = (280.4664567 + 360007.6982779 * jme + 0.03032028 * jme**2
             + jme**3 / 49931 - jme**4 / 15300 - jme**5 / 2000000)
        return M

    @staticmethod
    def equation_of_time(jme, alpha, delta_psi, e):
        """
        Equation of time

        Parameters
        ---------
        jme : ndarray
            Julian ephemeris millennium
        alpha : ndarray
            geocentric sun right ascension
        delta_psi : ndarray
            nutation longitude
        e : ndarray
            True ecliptic obliquity

        Returns
        -------
        E : ndarray
            Equation of time values for all timesteps
        """
        M = SPA.sun_mean_longitude(jme)
        E = (M - 0.0057183 - alpha + delta_psi * np.cos(np.radians(e)))
        # limit between 0 and 360
        E = E % 360
        # convert to minutes
        E *= 4
        greater = E > 20
        less = E < -20
        other = (E <= 20) & (E >= -20)
        E = greater * (E - 1440) + less * (E + 1440) + other * E
        return E

    def heliocentric_radius_vector(self, jme):
        """
        Compute heliocentric radius

        Parameters
        ----------
        jme : ndarray
            Julian ephemeris millennium

        Returns
        -------
        r : ndarray
            Heliocentric radius in radians
        """
        r = self._helicocentric_vector(self.HELIO_RADIUS_TABLE, jme)
        return r

    def centric_longitude(self, jme):
        """
        Compute the heliocentric and geocentric longitude

        Parameters
        ----------
        jme : ndarray

        Returns
        -------
        helio_lon : ndarray
            Heliocentric longitude for all timesteps
        geo_lon : ndarray
            Geocentric longitude for all timesteps
        """
        helio_lon = self.heliocentric_longitude(self.HELIO_LONG_TABLE, jme)
        geo_lon = self.geocentric_longitude(helio_lon)
        return helio_lon, geo_lon

    def centric_latitude(self, jme):
        """
        Compute the heliocentric and geocentric latitude

        Parameters
        ----------
        jme : ndarray

        Returns
        -------
        helio_lat : ndarray
            Heliocentric latitude for all timesteps
        geo_lat : ndarray
            Geocentric latitude for all timesteps
        """
        helio_lat = self.heliocentric_longitude(self.HELIO_LAT_TABLE, jme)
        geo_lat = self.geocentric_longitude(helio_lat)
        return helio_lat, geo_lat

    def longitude_nutation(self, jce):
        """
        Compute nutation longitude for all timesteps

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        delta_psi : ndarray
            nutation longitude
        """
        nut_arr = self.nutation_coefficients(jce)
        argsin = np.sin(np.radians(np.dot(self.NUTATION_YTERM_ARRAY, nut_arr)))

        a = self.NUTATION_ABCD_ARRAY[:, 0]
        b = self.NUTATION_ABCD_ARRAY[:, 1]

        delta_psi = np.sum((a + b * np.expand_dims(jce, axis=0).T).T * argsin,
                           axis=0)
        delta_psi = delta_psi / 36000000
        return delta_psi

    def obliquity_nutation(self, jce):
        """
        Compute nutation longitude for all timesteps

        Parameters
        ----------
        jce : ndarray
            Julian ephemeris century for all timesteps

        Returns
        -------
        delta_eps : ndarray
            nutation obliquity
        """
        nut_arr = self.nutation_coefficients(jce)
        argcos = np.cos(np.radians(np.dot(self.NUTATION_YTERM_ARRAY, nut_arr)))

        c = self.NUTATION_ABCD_ARRAY[:, 2]
        d = self. NUTATION_ABCD_ARRAY[:, 3]

        delta_eps = np.sum((c + d * np.expand_dims(jce, axis=0).T).T * argcos,
                           axis=0)
        delta_eps = delta_eps / 36000000
        return delta_eps
