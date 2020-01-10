import modis
import ims


class AlbedoError(Exception):
    pass


class CalcCompositeAlbedo:
    """
    Calculate composite albedo from MODIS and IMS for an arbitrary period of
    time.
    """
    @classmethod
    def run(cls, start_date, end_date, modis_path, ims_path, albedo_path):
        """
        Parameters
        ----------
        start_date : datetime instance
            Beginning of time frame to calculate albedo for
        start_date : datetime instance
            Beginning of time frame to calculate albedo for
        modis_path : str
            Path for MODIS data files
        ims_path : str
            Path for IMS data files
        albedo_path : str
            Path for composite albedo data files (output)
        """
        cca = cls(start_date, end_date, modis_path, ims_path, albedo_path)
        cca.date_range = cca.calc_date_range()
        cca.download_modis()
        cca.download_ims()
        cca.fill_ims_gaps()

        # Calculate mapping between IMS and MODIS
        nn_indicies = cca.calc_indicies()

        for date in cca.date_range:
            CompositeAlbedoDay.run(date, modis_path, ims_path, albedo_path,
                                   nn_indicies)

    def _init__(self, start_date, end_date, modis_path, ims_path, albedo_path):
        """ See parameter definitions in self.run() """
        self.start_date = start_date
        self.end_date = end_date
        self.modis_path = modis_path
        self.ims_path = ims_path
        self.albedo_path = albedo_path

    def calc_date_range(self):
        """ Calculate and return date range for analysis """
        pass

    def download_modis(self):
        """ Download MODIS data for date range """
        pass

    def download_ims(self):
        """ Download IMS data for date range """
        pass

    def fill_ims_gaps(self):
        """ Fill any temporal gaps in IMS data """
        pass


class CompositeAlbedoDay:
    @classmethod
    def run(cls, date, modis_path, ims_path, albedo_path, nn_indicies):
        """
        Merge MODIS and IMS data for one day.

        Parameters
        ----------
        date : datetime instance
            Date to calculate composite albedo for
        modis_path : str
            Path for MODIS data files
        ims_path : str
            Path for IMS data files
        albedo_path : str
            Path for composite albedo data files (output)
        nn_indices : numpy array
            Indices mapping IMS data to MODIS data
        """

        cad = cls(date, modis_path, ims_path, albedo_path, nn_indicies)
        cad.modis = cad.load_modis()
        # cad.modis.plot()
        cad.ims = cad.load_ims()
        albedo = cad.calc_albedo()
        cad.write_albedo(albedo)

        # TODO delete below
        return cad


    def __init__(self, date, modis_path, ims_path, albedo_path, nn_indicies):
        """ See parameter definitions for self.run() """
        self.date = date
        self.modis_path = modis_path
        self.ims_path = ims_path
        self.albedo_path = albedo_path
        self.nn_indicies = nn_indicies

        self.modis = None  # ModisDay object
        self.ims = None  # ImsDay object

    def load_modis(self):
        """ Load and return ModisDay instance """
        print(f'loading {self.date} from modis')
        # ModisDay is responsible for getting the closest day
        return modis.ModisDay(self.date, self.modis_path)

    def load_ims(self):
        """ Load and return ImsDay instance """
        print(f'loading {self.date} from ims')
        # TODO fix resolution handling
        return ims.ImsDay(self.date, self.ims_path, '1km')

    def calc_albedo(self):
        """ Calculate and return composite albedo as Albedo instance"""
        if self.modis is None or self.ims is None:
            raise AlbedoError('MODIS/IMS data must be loaded before running' +
                              ' calc_albedo()')
        print(f'Calculating composite albedo for {self.date}')

        # Find IMS pixels w/ snow: 3 = sea ice, 4 = snow on land
        snow_ind = (self.ims.data == 3) | (self.ims.data == 4)
        snow_ind_flat = snow_ind.flatten()
        self.snow = self.ims.data[snow_ind]
        self.snow_lat = self.ims.lat[snow_ind_flat]
        self.snow_lon = self.ims.lon[snow_ind_flat]
        print (f'Out of {self.ims.data.flatten().shape[0]:,} sites, ' + \
               f'{self.snow.shape[0]:,} have snow')

    @staticmethod
    def write_albedo(albedo):
        """
        Write albedo data to self.albedo_path.

        Parameters
        ----------
        albedo : Albedo instance
            Albedo data from self.calc_albedo()
        """
        pass


class Albedo:
    """ Holds composite albedo data from MODIS and IMS """
    pass
