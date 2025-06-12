import os
import shutil
import whakaari
from datetime import datetime, timedelta
from multiprocessing import Pool
from time import sleep
from typing import List

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
from obspy.signal.filter import bandpass
from scipy.integrate import cumtrapz

from .utils import to_datetime, load_dataframe, find_outliers, compute_rsam, compute_dsar, save_dataframe

STATIONS = {
    'WIZ': {
        'client_name': "GEONET",
        'nrt_name': 'https://service-nrt.geonet.org.nz',
        'channel': 'HHZ',
        'network': 'NZ',
        'location': '*',
    },
    'KRVZ': {
        'client_name': "GEONET",
        'nrt_name': 'https://service-nrt.geonet.org.nz',
        'channel': 'EHZ',
        'network': 'NZ',
        'location': '*',
    },
    'FWVZ': {
        'client_name': "GEONET",
        'nrt_name': 'https://service-nrt.geonet.org.nz',
        'channel': 'HHZ',
        'network': 'NZ',
        'location': '*',
    },
    'PVV': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'PV6': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'OKWR': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'VNSS': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'SSLW': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'REF': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
    'BELO': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'HHZ',
        'network': 'YC',
        'location': '*',
    },
    'CRPO': {
        'client_name': "IRIS",
        'nrt_name': 'https://service.iris.edu',
        'channel': 'HHZ',
        'network': 'OV',
        'location': '*',
    },
    'IVGP': {
        'client_name': 'https://webservices.ingv.it',
        'nrt_name': 'https://webservices.ingv.it',
        'channel': 'HHZ',
        'network': 'IV',
        'location': '*',
    },
    'AUS': {
        'client_name': 'IRIS',
        'nrt_name': 'https://service.iris.edu',
        'channel': 'EHZ',
        'network': 'AV',
        'location': '*',
    },
}


class TremorData:
    RATIO_NAMES = ['vlar', 'lrar', 'rmar', 'dsar']
    BAND_NAMES = ['vlf', 'lf', 'rsam', 'mf', 'hf']

    def __init__(self,
                 station: str,
                 parent: str = None,
                 data_dir: str = None,
                 eruptive_file: str = None,
                 n_jobs: int = 2,
                 cleanup_tmp_dir: bool = False,
                 verbose: bool = False, ):
        print(f'Version: {whakaari.__version__}')

        self.verbose = verbose
        self.station = station
        self.parent = parent
        self._stations = STATIONS

        self.data_dir = data_dir
        if data_dir is None:
            self.data_dir = os.getcwd()

        self.n_jobs = n_jobs

        # Originally self.file
        self.tremor_file = os.path.join(data_dir, 'input', f'{self.station}_tremor_data.csv')
        self.tremor_file_exists = os.path.isfile(self.tremor_file)

        self.eruptive_file = eruptive_file
        if eruptive_file is None:
            self.eruptive_file = os.path.join(data_dir, 'input', f'{self.station}_eruptive_periods.txt')
        assert os.path.isfile(self.eruptive_file), f'{self.eruptive_file} does not exist'

        self.cols = (self.RATIO_NAMES +
                     [f"{ratio_name}F" for ratio_name in self.RATIO_NAMES] +
                     self.BAND_NAMES + [f'{band_name}F' for band_name in self.BAND_NAMES])

        self.tes: List[datetime] = []
        self.df: pd.DataFrame = pd.DataFrame()

        # Originally self.ti
        self.datetime_start = None

        # Originally self.tf
        self.datetime_end = None

        self.tmp_dir = os.path.join(os.getcwd(), '_tmp')
        self.cleanup_tmp_dir = cleanup_tmp_dir

        if verbose:
            print(f'Station: {self.station}')
            print(f'Parent: {self.parent}')
            print(f'Data Dir: {self.data_dir}')
            print(f'Tremor Data file: {self.tremor_file}')
            print(f'Eruption file: {self.eruptive_file}')
            print(f'Number of jobs: {self.n_jobs}')
            print(f'Cols: {self.cols}')

        self.client = FDSNClient()
        self.client_nrt = FDSNClient()
        self._validate()

    def __repr__(self):
        if self.tremor_file_exists:
            return (f"TremorData('{self.station}', {self.parent}, {self.data_dir}, {self.eruptive_file}, "
                    f"{self.n_jobs}, {self.verbose})")
        else:
            return 'no data'

    @property
    def stations(self):
        return self._stations

    @stations.setter
    def stations(self, stations: dict):
        self._stations = stations

    def update(self, datetime_start: str = None, datetime_end: str = None, n_jobs: int = None):
        """ Return tremor data in requested date range.

        :param datetime_start: start date of tremor data
        :param datetime_end: end date of tremor data
        :param n_jobs: number of parallel jobs
        :return: DataFrame with tremor data

        :type datetime_start: str
        :type datetime_end: str
        :type n_jobs: int
        :rtype: pd.DataFrame
        """
        os.makedirs(self.tmp_dir, exist_ok=True)

        if datetime_start is None:
            if self.datetime_end is not None:
                datetime_start = datetime(self.datetime_end.year, self.datetime_end.month,
                                          self.datetime_end.day, 0, 0, 0)
            else:
                datetime_start = self._probe_start()

        datetime_end = datetime_end or datetime.today() + timedelta(days=1.)
        datetime_start_obj = to_datetime(datetime_start)
        datetime_end_obj = to_datetime(datetime_end)

        n_days = (datetime_end_obj - datetime_start_obj).days

        parallels = [[index, datetime_start_obj, self.station] for index in range(n_days)]
        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        if self.verbose:
            print(f'Parallels :')
            print(parallels)

        if n_jobs == 1:
            print('=' * 60)
            print(f'Station {self.station}: Downloading data in serial')
            print('=' * 60)
            for parallel in parallels:
                progress = str(parallel[0] + 1) + '/' + str(len(parallels))
                print(f'⌚ {progress}')
                self.get_data_for_day(*parallel)
        else:
            print(f'Station {self.station}: Downloading data in parallel')
            print('From: ' + str(datetime_start_obj))
            print('To: ' + str(datetime_end_obj))
            print('\n')
            p = Pool(n_jobs)
            p.starmap(self.get_data_for_day, parallels)
            p.close()
            p.join()

        # Read temporary files in as dataframes for concatenation with existing data
        cols = self.cols
        dfs = [self.df[cols]]
        for index_file in range(n_days):
            filepath = os.path.join(self.tmp_dir, '_tmp/_tmp_fl_{:05d}.csv'.format(index_file))
            if not os.path.isfile(filepath):
                continue

            dfs.append(load_dataframe(filepath, index_col=0, parse_dates=True, infer_datetime_format=True))

        if self.cleanup_tmp_dir:
            if self.verbose:
                print(f'Cleaning up temp dir: {self.tmp_dir}')
            shutil.rmtree(self.tmp_dir)

        self.df = pd.concat(dfs)

        # Impute missing data using linear interpolation and save file
        df = self.df.loc[~self.df.index.duplicated(keep='last')]
        filename, filetype = self.tremor_file.split('\\')[-1].split('.')
        save_path = os.path.join(os.getcwd(), f'{filename}_nitp.{filetype}')
        save_dataframe(df, save_path, index=True)

        if self.verbose:
            print(f'Imputing dataframe saved to : {save_path}')

        df.index = pd.to_datetime(df.index)
        self.df = df.resample('10T').interpolate('linear')

        save_interpolate_path = os.path.join(os.getcwd(), f'{filename}.{filetype}')
        save_dataframe(self.df, save_interpolate_path, index=True)
        self.datetime_start = self.df.index[0]
        self.datetime_end = self.df.index[-1]

    def _probe_start(self):
        if self.verbose:
            print(f'_probe_start :: Tries to figure out when the first available data for a station {self.station}')

        station = self.stations[self.station]

        try:
            if self.verbose:
                print(f'_probe_start :: Downloading from FDSN')
            client = FDSNClient(station['client_name'])
            site = client.get_stations(station=self.station, level="response", channel=station['channel'])
            return site.networks[0].stations[0].start_date
        except ConnectionError as e:
            raise ConnectionError(f'_probe_start :: Unable to connect to FDSN with error {e}')

    def get_data_from_stream(self, stream, inventory=None) -> np.ndarray:
        if len(stream.traces) == 0:
            raise ValueError('get_data_from_stream :: Stream has no traces')
        elif len(stream.traces) > 1:
            try:
                stream.merge(fill_value='interpolate').traces[0]
            except Exception as e:
                if self.verbose:
                    print('get_data_from_stream :: Failed to merge traces. Try to interpolate.')
                stream = stream.interpolate(100).merge(fill_value='interpolate').traces[0]

        if inventory is not None:
            stream.remove_sensitivity(inventory=inventory)

        return stream.traces[0].data

    def get_data_for_day(self, index: int, date: datetime, station: str):
        _date = UTCDateTime(date)
        day_second = 24 * 3600
        freq_bands = [[0.01, 0.1], [0.1, 2], [2, 5], [4.5, 8], [8, 16]]
        band_names = self.BAND_NAMES
        frs = [200, 200, 200, 100, 50]

        frequency = 100
        decimation = 1
        _station = self.stations[station]

        if self.verbose:
            print(f"✍️ Checking FDSN Client connection using {_station['client_name']} and {_station['nrt_name']}")

        attempts = 0
        while attempts < 10:
            try:
                self.client = FDSNClient(_station['client_name'])
                self.client_nrt = FDSNClient(_station['nrt_name'])
            except FDSNException:
                if attempts > 9:
                    raise FDSNException('get_data_for_day :: Timed out after 10 attempts, couldn\'t '
                                        'connect to FDSN service')

                print(f'⚠️ Attempt no {attempts}: Reconnecting.. after 15s')
                sleep(15)
                attempts += 1
            else:
                attempts = 10

        print(f'✅ Connected')
        client = self.client
        client_nrt = self.client_nrt

        try:
            site = client.get_stations(starttime=_date + (index * day_second), endtime=_date + ((index + 1) * day_second),
                                       station=station, level="response", channel=_station['channel'])

        except (FDSNNoDataException, FDSNException) as e:
            print(f'⚠️ Failed to download inventory')
            site = None

        if site is not None:
            print(f'🛖 Inventory Downloaded')

        pad_f = 0.1
        try:
            print(f'⌛ Downloading using Client')

            network = _station['network']
            location = _station['location']
            channel = _station['channel']
            start_date = _date + ((index - pad_f) * day_second)
            end_date = _date + ((index + 1 + pad_f) * day_second)

            # print(network, station, location, channel, start_date, end_date)

            st = client.get_waveforms(network, station, location, channel,
                                      start_date, end_date)

            download_dir = os.path.join(os.getcwd(), 'download')
            os.makedirs(download_dir, exist_ok=True)

            date_str = date.strftime('%Y-%m-%d')

            st.write(os.path.join(download_dir, f'{network}_{station}_{date_str}_{index}'), format='MSEED')

            data = self.get_data_from_stream(st, site)
            if data is None:
                print(f'❌ Data not found.')
                return
            else:
                print(f'✅ Stream downloaded')
            # if less than 1 day of data, try different client
            _len = 600 * frequency
            if len(data) < _len:
                raise FDSNNoDataException(f'❌ get_data_for_day :: Data length less than {_len}')

        except (ValueError, ObsPyMSEEDFilesizeTooSmallError, FDSNNoDataException, FDSNException) as e:
            print(f'⌛ Downloading using Client Failed. Try to use NRT')
            try:
                st = client_nrt.get_waveforms(_station['network'], station, _station['location'], _station['channel'],
                                              _date + (index - pad_f) * day_second,
                                              _date + (index + 1 + pad_f) * day_second)
                data = self.get_data_from_stream(st, site)
            except (FDSNNoDataException, ValueError, FDSNException) as e:
                raise ConnectionError(f'❌ Failed to download using NRT :: {e}')

        if decimation > 1:
            st.decimate(decimation)
            frequency = frequency // decimation

        data = st.traces[0]

        i0 = int((_date + (index * day_second) - st.traces[0].meta['starttime']) * frequency) + 1

        if i0 < 0 or i0 >= len(data):
            if self.verbose:
                print(
                    f'get_data_for_day :: The data length is not valid. i0 value is {i0} '
                    f'and data length is {len(data)}')
            return None

        i1 = int(24 * 3600 * frequency)
        if (i0 + i1) > len(data):
            i1 = len(data)
        else:
            i1 += i0

        # Process frequency bands
        data_i = cumtrapz(data, dx=1. / frequency, initial=0)
        data_i -= data_i[i0]

        start_time = st.traces[0].meta['starttime'] + timedelta(seconds=(i0 + 1) / frequency)

        # Round start time to nearest 10 min increment
        start_time_day = UTCDateTime(f"{start_time.year}-{start_time.month}-{start_time.day} 00:00:00")

        start_time = start_time_day + int(np.round((start_time - start_time_day) / 600)) * 600

        n = 600 * frequency  # 10 minutes windows in seconds
        m = (i1 - i0) // n  # number of windows

        # Apply filters and remove filter response
        _datas = []
        _data_is = []
        for (freq_min, freq_max), fr in zip(freq_bands, frs):
            _data = abs(bandpass(data, freq_min, freq_max, frequency)[i0:i1]) * 1.e9
            _data_i = abs(bandpass(data_i, freq_min, freq_max, frequency)[i0:i1]) * 1.e9
            _datas.append(_data)
            _data_is.append(_data_i)

        # Find outliers in each 10-min window
        outliers, max_idxs = find_outliers(_datas, n, m)

        datas = []
        columns = []
        asymmetry_factor = 0.1  # Asymmetry factor
        number_sub_domains = 4
        sub_domain_range = n // number_sub_domains  # No. data points per subDomain

        # Compute rsam and other bands (w/ EQ filter)
        data_rsam, columns_rsam = compute_rsam(_datas, band_names=band_names, m=m, n=n, outliers=outliers,
                                               max_idxs=max_idxs, asymmetry_factor=asymmetry_factor,
                                               sub_domain_range=sub_domain_range)
        datas.append(data_rsam)
        columns.append(columns_rsam)

        # Compute dsar (w/ EQ filter)
        data_dsar, column_dsar = compute_dsar(_data_is, ratio_names=self.RATIO_NAMES, m=m, n=n, outliers=outliers,
                                              max_idxs=max_idxs, asymmetry_factor=asymmetry_factor,
                                              sub_domain_range=sub_domain_range)
        datas.append(data_dsar)
        columns.append(column_dsar)

        # Write out temporary file
        datas = np.array(datas)
        time = [(start_time + datas_index * 600).datetime for datas_index in range(datas.shape[1])]
        df = pd.DataFrame(zip(*datas), columns=columns, index=pd.Series(time))

        csv_path = os.path.join(self.tmp_dir, '_tmp_fl_{:05d}.csv'.format(index))
        save_dataframe(df, csv_path, index=True, index_label='time')

    def _validate(self):
        """
        Load existing file and check date range of data.
        """
        with open(self.eruptive_file, 'r') as fp:
            self.tes = [to_datetime(_line.rstrip()) for _line in fp.readlines()]

        # Check if tremor data file exists. If not, create one
        if not self.tremor_file_exists:
            if self.verbose:
                print(f'Creating new tremor data...')
            df = pd.DataFrame(columns=self.cols)
            df.to_csv(self.tremor_file, index_label='time')
            if self.verbose:
                print(f'New tremor data saved to: {self.tremor_file}')

        # Load dataframe
        self.df = load_dataframe(self.tremor_file, index_col=0, parse_dates=True, infer_datetime_format=True)

        if len(self.df) > 0:
            self.datetime_start = self.df.index[0]
            self.datetime_end = self.df.index[-1]
            if self.verbose:
                print(f'Dataframe loaded: {self.df.shape}')
                print(f'Start date: {self.datetime_start}')
                print(f'End date: {self.datetime_end}')
