# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:50:16 2019

@author: gbuster
"""

import os
import pandas as pd
import numpy as np
import h5py


def get_table(d):
    """Get one year of data from the directory.

    Parameters
    ----------
    d : str
        Directory containing surfrad data files.

    Returns
    -------
    annual_table : list
        List of data rows from all files in d. First entry is the list of
        column headers.
    """

    # get list of data files
    flist = os.listdir(d)

    # iterate through data files
    for i, fname in enumerate(flist):

        # get readlines iterator
        with open(os.path.join(d, fname), 'r') as f:
            lines = f.readlines()

        # iterate through lines
        for j, line in enumerate(lines):

            # reduce multiple spaces to a single space, split columns
            while '  ' in line:
                line = line.replace('  ', ' ')
            cols = line.strip(' ').split(' ')

            # Set table header or append data to table
            if j == 0:
                table = [cols]
            else:
                table.append(cols)

        # upon finishing table concatenation, initialize annual table
        # or append to annual table
        if i == 0:
            annual_table = table
        else:
            if table[0] == annual_table[0]:
                # make sure headers are the same
                annual_table += table[1:]
            else:
                msg = ('Headers for "{}" does not match annual table '
                       'headers: {}'
                       .format(os.path.join(d, fname), annual_table[0]))
                raise ValueError(msg)
    return annual_table


def extract_all(root_dir, dir_out, years=range(1998, 2018),
                site_codes=('bon', 'dra', 'fpk', 'gwn', 'psu', 'sxf', 'tbl')):
    """Extract all surfrad measurement data into h5 files.

    Parameters
    ----------
    root_dir : str
        Root directory containing surfrad ground measurement data.
        Directory containing data files is:
            /root_dir/site_code/year/
    dir_out : str
        Target path to save surfrad h5 data.
    years : iterable
        Years to process for.
    site_codes : list | tuple
        Sites codes that makeup directory names.
    """

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for site in site_codes:
        for year in years:

            # look for target data directory
            d = os.path.join(root_dir, site, str(year))

            # set target output filename
            fout = '{}_{}.h5'.format(site, year)

            if not os.path.exists(d):
                print('Skipping: "{}" for {}'.format(site, year))

            else:
                annual_table = get_table(d)

                headers = [h.lower() for h in annual_table[0]]
                df = pd.DataFrame(annual_table[1:], columns=headers)
                mapping = {'swdn': 'ghi',
                           'dirsw': 'dni',
                           'difsw': 'dhi'}
                df = df[['zdate', 'ztim', 'swdn', 'dirsw', 'difsw']]
                df = df.rename(mapping, axis='columns')
                df['time_string'] = df['zdate'] + ' ' + df['ztim'].str.zfill(4)
                ti = pd.to_datetime(df['time_string'], format='%Y%m%d %H%M')
                df.index = ti

                with h5py.File(os.path.join(dir_out, fout), 'w') as f:

                    time_index = np.array(df.index.astype(str), dtype='S20')
                    ds = f.create_dataset('time_index', shape=time_index.shape,
                                          dtype=time_index.dtype, chunks=None)
                    ds[...] = time_index

                    for dset in ['dhi', 'dni', 'ghi']:
                        df[dset] = np.round(df[dset].astype(float))\
                            .astype(np.int16)
                        ds = f.create_dataset(dset, shape=df[dset].shape,
                                              dtype=df[dset].dtype,
                                              chunks=None)
                        ds[...] = df[dset].values


if __name__ == '__main__':
    root_dir = '/projects/PXS/surfrad_radflux'
    dir_out = '/home/gbuster/surfrad_data'
    extract_all(root_dir, dir_out)
