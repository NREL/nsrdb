"""
Script to download MODIS V6.1 data for NSRDB albedo processing
for a range of dates to the current folder. The user will need a NASA
Earthdata login and will need to setup the .netrc and .usr_cookies files as
described at:

wiki.earthdata.nasa.gov/display/EL/How+To+Access+Data+With+cURL+And+Wget

Mike Bannister
2/27/2020
"""

import os
import sys
from datetime import datetime as dt
from datetime import timedelta

cmd = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies '
cmd += '--keep-session-cookies -O "{outfile}" "{line}"'
url = 'https://e4ftl01.cr.usgs.gov/MOTA/MCD43GF.061/'


def main(year):
    """Download MODIS v6.1 data"""
    start = dt(year, 1, 1)
    end = dt(year, 12, 31)

    data_dir = 'v6.1/source_{}'.format(year)
    if not os.path.isdir(data_dir):
        print('Making dir {}'.format(data_dir))
        os.mkdir(data_dir)

    for date in daterange(start, end):
        year = date.year
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        doy = str(date.timetuple().tm_yday).zfill(3)

        # E.g. '2018.03.03/MCD43GF_wsa_shortwave_063_2018_V061.hdf'
        path = f'{year}.{month}.{day}/'
        _file = f'MCD43GF_wsa_shortwave_{doy}_{year}_V061.hdf'

        outfile = data_dir + '/' + _file
        if not os.path.exists(outfile):
            day_cmd = cmd.format(outfile=outfile, line=url + path + _file)
            print(day_cmd)
            os.system(day_cmd)


def daterange(start_date, end_date):
    """
    Create a range of dates.

    From https://stackoverflow.com/questions/1060279/
    iterating-through-a-range-of-dates-in-python
    """
    for n in range(int((end_date - start_date).days) + 1):
        yield start_date + timedelta(n)


if __name__ == '__main__':
    start_year = int(sys.argv[1])
    end_year = int(sys.argv[2])
    for year in range(start_year, end_year + 1):
        print('Grabbing modis for', year)
        main(year)
