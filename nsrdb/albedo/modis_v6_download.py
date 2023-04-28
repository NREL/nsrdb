"""
This script to downloads MODIS V6 data for NSRDB albedo processing
for a range of dates to the current folder. The user will need a NASA
Earthdata login and will need to setup the .netrc and .usr_cookies files as
described at:

https://wiki.earthdata.nasa.gov/display/
            EL/How+To+Access+Data+With+cURL+And+Wget

Mike Bannister
2/27/2020
"""
from datetime import datetime as dt
from datetime import timedelta
import os


def main():
    """
    Download MODIS V6 data
    """
    start = dt(2017, 1, 1)
    end = dt(2017, 12, 31)

    curl_cmd = 'curl -O -b ~/.urs_cookies -c ~/.urs_cookies -L -n '
    url = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD43GF.006/'

    for date in daterange(start, end):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        doy = str(date.timetuple().tm_yday).zfill(3)

        # E.g. '2000.03.03/MCD43GF_wsa_shortwave_063_2000_V006.hdf'
        path = f'{year}.{month}.{day}/'
        _file = f'MCD43GF_wsa_shortwave_{doy}_{year}_V006.hdf'

        cmd = curl_cmd + url + path + _file
        print(cmd)
        os.system(cmd)


def daterange(start_date, end_date):
    """
    Create a range of dates.

    From https://stackoverflow.com/questions/1060279/
    iterating-through-a-range-of-dates-in-python
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


if __name__ == '__main__':
    main()
