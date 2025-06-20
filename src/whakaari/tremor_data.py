from __future__ import annotations

import os
import shutil
import whakaari
from datetime import datetime, timedelta
from multiprocessing import Pool
from time import sleep
from typing import List

import numpy as np
import pandas as pd
from obspy import UTCDateTime, Stream, read
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.header import FDSNException, FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDFilesizeTooSmallError
from obspy.signal.filter import bandpass
from scipy.integrate import cumtrapz
from scipy.signal import stft

from .utils import (
    to_datetime,
    load_dataframe,
    find_outliers,
    compute_rsam,
    compute_dsar,
    save_dataframe,
)

STATIONS = {
    "WIZ": {
        "client_name": "GEONET",
        "nrt_name": "https://service.geonet.org.nz",
        "channel": "HHZ",
        "network": "NZ",
        "location": "*",
    },
    "KRVZ": {
        "client_name": "GEONET",
        "nrt_name": "https://service-nrt.geonet.org.nz",
        "channel": "EHZ",
        "network": "NZ",
        "location": "*",
    },
    "FWVZ": {
        "client_name": "GEONET",
        "nrt_name": "https://service-nrt.geonet.org.nz",
        "channel": "HHZ",
        "network": "NZ",
        "location": "*",
    },
    "PVV": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "PV6": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "OKWR": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "VNSS": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "SSLW": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "REF": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "BELO": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "HHZ",
        "network": "YC",
        "location": "*",
    },
    "CRPO": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "HHZ",
        "network": "OV",
        "location": "*",
    },
    "IVGP": {
        "client_name": "https://webservices.ingv.it",
        "nrt_name": "https://webservices.ingv.it",
        "channel": "HHZ",
        "network": "IV",
        "location": "*",
    },
    "AUS": {
        "client_name": "IRIS",
        "nrt_name": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
}


class TremorData:
    RATIO_NAMES = ["vlar", "lrar", "rmar", "dsar"]
    BAND_NAMES = ["vlf", "lf", "rsam", "mf", "hf"]

    def __init__(
        self,
        station: str,
        parent=None,
        data_dir: str = None,
        eruptive_file: str = None,
        tremor_data_file: str = None,
        n_jobs: int = 2,
        cleanup_tmp_dir: bool = False,
        verbose: bool = False,
    ):
        print(f"Version: {whakaari.__version__}")

        self.verbose = verbose
        self.station = station
        self.parent = parent

        self._stations = STATIONS
        self._freq_bands: List[List[float]] = [
            [0.01, 0.1],
            [0.1, 2],
            [2, 5],
            [4.5, 8],
            [8, 16],
        ]
        self._channel = "*"

        self.data_dir = data_dir
        if data_dir is None:
            self.data_dir = os.getcwd()

        self.n_jobs = n_jobs

        # Originally self.file
        if tremor_data_file is None:
            tremor_data_file = os.path.join(
                data_dir, "output", f"{self.station}_tremor_data.csv"
            )
        self.tremor_file = tremor_data_file
        self.tremor_file_exists = os.path.exists(self.tremor_file)

        self.eruptive_file = eruptive_file
        if eruptive_file is None:
            self.eruptive_file = os.path.join(
                data_dir, "input", f"{self.station}_eruptive_periods.txt"
            )
        assert os.path.isfile(
            self.eruptive_file
        ), f"{self.eruptive_file} does not exist"

        self.cols = (
            self.RATIO_NAMES
            + [f"{ratio_name}F" for ratio_name in self.RATIO_NAMES]
            + self.BAND_NAMES
            + [f"{band_name}F" for band_name in self.BAND_NAMES]
        )

        self.tes: List[datetime] = []
        self.df: pd.DataFrame = pd.DataFrame()

        # Originally self.ti
        self.datetime_start = None

        # Originally self.tf
        self.datetime_end = None

        self.tmp_dir = os.path.join(os.getcwd(), "_tmp", station)
        self.cleanup_tmp_dir = cleanup_tmp_dir

        if verbose:
            print(f"Station: {self.station}")
            print(f"Parent: {self.parent}")
            print(f"Data Dir: {self.data_dir}")
            print(f"Tremor Data file: {self.tremor_file}")
            print(f"Tremor Data exists: {self.tremor_file_exists}")
            print(f"Eruption file: {self.eruptive_file}")
            print(f"Number of jobs: {self.n_jobs}")
            print(f"Cols: {self.cols}")

        self.client = FDSNClient()

        # NRT = Near Real-time
        self.client_nrt = FDSNClient()
        self._validate()

    def __repr__(self):
        if self.tremor_file_exists:
            return (
                f"TremorData('{self.station}', {self.parent}, {self.data_dir}, {self.eruptive_file}, "
                f"{self.n_jobs}, {self.verbose})"
            )
        else:
            return "no data"

    @property
    def freq_bands(self) -> List[List[float]]:
        return self._freq_bands

    @freq_bands.setter
    def freq_bands(self, freq_bands: List[List[float]]):
        self._freq_bands = freq_bands

    @property
    def channel(self) -> str:
        return self._channel

    @channel.setter
    def channel(self, channel: str):
        self._channel = channel

    @property
    def stations(self):
        return self._stations

    @stations.setter
    def stations(self, stations: dict):
        self._stations = stations

    def update(
        self,
        datetime_start: str = None,
        datetime_end: str = None,
        n_jobs: int = None,
        sds_dir: str = None,
    ):
        """Return tremor data in the requested date range.

        :param datetime_start: Start date of tremor data
        :param datetime_end: end date of tremor data
        :param n_jobs: number of parallel jobs
        :sds_dir: directory to save tremor data
        :return: DataFrame with tremor data

        :type datetime_start: str
        :type datetime_end: str
        :type n_jobs: int
        :type sds_dir: str
        :rtype: pd.DataFrame
        """
        os.makedirs(self.tmp_dir, exist_ok=True)

        if datetime_start is None:
            if self.datetime_end is not None:
                _datetime_end = to_datetime(self.datetime_end)
                datetime_start = datetime(
                    _datetime_end.year,
                    _datetime_end.month,
                    _datetime_end.day,
                    0,
                    0,
                    0,
                )
            else:
                datetime_start = self._probe_start()

        datetime_end = datetime_end or datetime.today() + timedelta(days=1.0)
        datetime_start_obj = to_datetime(datetime_start)
        datetime_end_obj = to_datetime(datetime_end)

        n_days = (datetime_end_obj - datetime_start_obj).days

        parallels = [
            [index, datetime_start_obj, self.station, sds_dir]
            for index in range(n_days)
        ]
        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        if n_jobs == 1:
            print("=" * 60)
            print(f"Station {self.station}: Downloading data in serial")
            for parallel in parallels:
                progress = str(parallel[0] + 1) + "/" + str(len(parallels))
                print("=" * 60)
                print(f"⌚ Progress :: {progress}")
                self.get_data_for_day(*parallel)
        else:
            print("=" * 60)
            print(f"Station {self.station}: Downloading data in parallel")
            print("From: " + str(datetime_start_obj))
            print("To: " + str(datetime_end_obj))
            print("=" * 60)
            p = Pool(n_jobs)
            p.starmap(self.get_data_for_day, parallels)
            p.close()
            p.join()

        # Read temporary files in as dataframes for concatenation with existing data
        dfs = []
        for index_file in range(n_days):
            filepath = os.path.join(
                self.tmp_dir, "_tmp_fl_{:05d}.csv".format(index_file)
            )
            if not os.path.isfile(filepath):
                if self.verbose:
                    print(f"⚠️ Temporary file not found in {filepath}")
                continue

            dfs.append(
                load_dataframe(
                    filepath, index_col=0, parse_dates=True, infer_datetime_format=True
                )
            )

        if self.cleanup_tmp_dir:
            if self.verbose:
                print(f"Cleaning up temp dir: {self.tmp_dir}")
            shutil.rmtree(self.tmp_dir)

        if len(dfs) == 0:
            raise ValueError(
                f"❌ update:: Cannot get data from temporary dir ({self.tmp_dir})"
            )

        if len(dfs) == 1:
            df = dfs[0]
        else:
            print(f"Length DFS :: {len(dfs)}")
            df = pd.concat(dfs, sort=False)

        # Impute missing data using linear interpolation and save a file
        df = df.loc[~df.index.duplicated(keep="last")]
        filename, filetype = self.tremor_file.split("\\")[-1].split(".")

        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, f"{filename}_nitp.{filetype}")
        save_dataframe(df, save_path, index=True)

        df.index = pd.to_datetime(df.index)
        self.df = df.resample("10T").interpolate("linear")
        save_interpolate_path = os.path.join(output_dir, f"{filename}.{filetype}")
        save_dataframe(self.df, save_interpolate_path, index=True)

        self.datetime_start = self.df.index[0]
        self.datetime_end = self.df.index[-1]

        if self.verbose:
            print(f"💾 Dataframe saved to : {save_interpolate_path}")
            print(f"📅 Start Date: {self.datetime_start}")
            print(f"📅 End Date: {self.datetime_end}")

    def _probe_start(self):
        if self.verbose:
            print(
                f"_probe_start :: Tries to figure out when the first available data for a station {self.station}"
            )

        station = self.stations[self.station]

        try:
            if self.verbose:
                print(f"_probe_start :: Downloading from FDSN")
            client = FDSNClient(station["client_name"])
            site = client.get_stations(
                station=self.station, level="response", channel=station["channel"]
            )
            return site.networks[0].stations[0].start_date
        except ConnectionError as e:
            raise ConnectionError(
                f"_probe_start :: Unable to connect to FDSN with error {e}"
            )

    def get_data_from_stream(self, stream, inventory=None) -> np.ndarray:
        if len(stream.traces) == 0:
            raise ValueError("get_data_from_stream :: Stream has no traces")
        elif len(stream.traces) > 1:
            try:
                stream.merge(fill_value="interpolate").traces[0]
            except Exception as e:
                if self.verbose:
                    print(
                        f"get_data_from_stream :: Failed to merge traces. Try to interpolate. {e}"
                    )
                stream = (
                    stream.interpolate(100).merge(fill_value="interpolate").traces[0]
                )

        if inventory is not None:
            stream.remove_sensitivity(inventory=inventory)

        return stream.traces[0].data

    def download(
        self,
        index: int,
        utc_datetime: UTCDateTime,
        station: str,
        frequency: float = 100.0,
    ) -> Stream:
        _station = self.stations[station]

        _date = utc_datetime
        day_second = 24 * 3600

        download_dir = os.path.join(os.getcwd(), "download")
        os.makedirs(download_dir, exist_ok=True)

        date_str = _date.strftime("%Y-%m-%d")
        filepath = os.path.join(
            download_dir, f"{_station['network']}_{station}_{date_str}_{index}"
        )

        if os.path.isfile(filepath):
            if self.verbose:
                print(f"✅ File exists :: {filepath}")
            return read(filepath, format="MSEED")

        if self.verbose:
            print(
                f"✍️ Checking FDSN Client connection using {_station['client_name']} from {_station['nrt_name']}"
            )

        attempts = 0
        while attempts < 10:
            try:
                self.client = FDSNClient(_station["client_name"])
                self.client_nrt = FDSNClient(_station["nrt_name"])
            except FDSNException:
                if attempts > 9:
                    raise FDSNException(
                        "get_data_for_day :: Timed out after 10 attempts, couldn't "
                        "connect to FDSN service"
                    )

                print(f"⚠️ Attempt no {attempts}: Reconnecting.. after 15s")
                sleep(15)
                attempts += 1
            else:
                attempts = 10

        print(f"✅ Connected")
        client = self.client
        client_nrt = self.client_nrt

        # Download instrument response
        try:
            inventory = client.get_stations(
                starttime=_date + (index * day_second),
                endtime=_date + ((index + 1) * day_second),
                station=station,
                level="response",
                channel=_station["channel"],
            )

        except (FDSNNoDataException, FDSNException) as e:
            print(f"⚠️ Failed to download inventory :: {e}")
            inventory = None

        if inventory is not None:
            print(f"🛖 Inventory Downloaded")

        pad_f = 0.01
        try:
            print(f"⌛ Downloading using Client")
            network = _station["network"]
            location = _station["location"]
            channel = _station["channel"]
            start_date = _date + ((index - pad_f) * day_second)
            end_date = _date + ((index + 1 + pad_f) * day_second)

            print(f"Datetime Start :: {start_date}")
            print(f"Datetime End :: {end_date}")

            st = client.get_waveforms(
                network, station, location, channel, start_date, end_date
            )

            data = self.get_data_from_stream(st, inventory)
        except (
            ValueError,
            ObsPyMSEEDFilesizeTooSmallError,
            FDSNNoDataException,
            FDSNException,
        ) as e:
            print(
                f"⌛ Downloading using Client Failed. Try to use NRT (Near Real-Time) Client. {e}"
            )
            try:
                st = client_nrt.get_waveforms(
                    _station["network"],
                    station,
                    _station["location"],
                    _station["channel"],
                    _date + (index - pad_f) * day_second,
                    _date + (index + 1 + pad_f) * day_second,
                )
                data = self.get_data_from_stream(st, inventory)
            except (FDSNNoDataException, ValueError, FDSNException) as e:
                raise ConnectionError(f"❌ Failed to download using NRT :: {e}")

        if data is None:
            print(f"❌ Data not found.")
            return Stream()

        # if less than 1 day of data, try a different client
        _len = 600 * frequency
        if len(data) < _len:
            raise FDSNNoDataException(
                f"❌ get_data_for_day :: Data length less than {_len}"
            )

        st.write(filepath, format="MSEED")
        print(f"✅ Stream saved to :: {filepath}")

        return st

    def download_from_sds(
        self, sds_dir: str, utc_datetime: UTCDateTime, station: str
    ) -> Stream:
        year = utc_datetime.year
        julian_day = utc_datetime.strftime("%j")
        stations = self.stations[station]
        network = stations["network"]
        location = stations["location"]
        channel = stations["channel"]
        channel_type = "D"

        filename = f"{network}.{station}.{location}.{channel}.{channel_type}.{year}.{julian_day}"

        # SDS Path
        miniseed_file = os.path.join(
            sds_dir,
            str(year),
            network,
            station,
            f"{channel}.{channel_type}",
            filename,
        )

        # Checking file exists
        if not os.path.isfile(miniseed_file):
            print(f"❌ File not found :: {miniseed_file}")
            return Stream()

        # Load miniseed
        st = read(miniseed_file, format="MSEED")
        return st

    def get_data_for_day(
        self, index: int, date: datetime, station: str, sds_dir: str = None
    ) -> None:
        # Skip for temporary file already exists
        csv_path = os.path.join(self.tmp_dir, "_tmp_fl_{:05d}.csv".format(index))
        if os.path.isfile(csv_path):
            if self.verbose:
                print(f"✅ Temp file exists :: {csv_path}")
            return None

        _date = UTCDateTime(date)
        day_second = 24 * 3600
        freq_bands = self.freq_bands
        band_names = self.BAND_NAMES
        frs = [200, 200, 200, 100, 50]

        frequency = 100
        decimation = 1

        # Download using FDSN
        if sds_dir is None:
            st = self.download(index, _date, station)
        else:
            # Need to be modifed
            st = self.download_from_sds(sds_dir, _date + timedelta(index), station)

        if len(st) == 0:
            return None

        # Pre-processing stream
        if self.verbose:
            print(f"📶 Pre-processing data, apply filter...")

        st = st.merge(method="interpolate")

        if decimation > 1:
            st.decimate(decimation)
            frequency = frequency // decimation

        trace = st.traces[0]

        # Checking data length
        i0 = (
            int(
                (_date + (index * day_second) - st.traces[0].meta["starttime"])
                * frequency
            )
            + 1
        )

        if i0 < 0 or i0 >= len(trace.data):
            if self.verbose:
                print(
                    f"get_data_for_day :: The data length is not valid. i0 value is {i0} "
                    f"and data length is {len(trace)}"
                )
            return None

        # Ensuring data length for one day downloaded is not over 24 * 3600 * frequency
        i1 = int(24 * 3600 * frequency)
        if (i0 + i1) > len(trace):
            i1 = len(trace)
        else:
            i1 += i0

        # Process frequency bands
        # Integrate velocity to displacement
        data_i = cumtrapz(trace, dx=1.0 / frequency, initial=0)
        data_i -= data_i[i0]

        start_time = st.traces[0].meta["starttime"]
        if sds_dir is None:
            start_time = st.traces[0].meta["starttime"] + timedelta(
                seconds=(i0 + 1) / frequency
            )

        print(f"Start time : {start_time}")

        # Round start time to the nearest 10 min increment
        start_time_day = UTCDateTime(
            f"{start_time.year}-{start_time.month}-{start_time.day} 00:00:00"
        )

        start_time = (
            start_time_day + int(np.round((start_time - start_time_day) / 600)) * 600
        )

        n = 60 * 10 * frequency  # Number of samples in 10 minutes
        m = (i1 - i0) // n  # number of windows

        # Apply filters and remove filter response
        _datas = []  # Trace data
        _data_is = []  # Displacement data
        for (freq_min, freq_max), fr in zip(freq_bands, frs):
            _data = abs(bandpass(trace, freq_min, freq_max, frequency)[i0:i1]) * 1.0e9
            _data_i = (
                abs(bandpass(data_i, freq_min, freq_max, frequency)[i0:i1]) * 1.0e9
            )
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
        if self.verbose:
            print(f"🧮 Computing RSAM ...", end="")

        data_rsam, columns_rsam = compute_rsam(
            _datas,
            band_names=band_names,
            m=m,
            n=n,
            outliers=outliers,
            max_idxs=max_idxs,
            asymmetry_factor=asymmetry_factor,
            sub_domain_range=sub_domain_range,
        )
        datas += data_rsam
        columns += columns_rsam
        if self.verbose:
            print(f" Done")

        # Compute dsar (w/ EQ filter)
        if self.verbose:
            print(f"🧮 Computing DSAR ...", end="")

        data_dsar, column_dsar = compute_dsar(
            _data_is,
            ratio_names=self.RATIO_NAMES,
            m=m,
            n=n,
            outliers=outliers,
            max_idxs=max_idxs,
            asymmetry_factor=asymmetry_factor,
            sub_domain_range=sub_domain_range,
        )
        datas += data_dsar
        columns += column_dsar
        if self.verbose:
            print(f" Done")

        # Write out a temporary file
        datas = np.array(datas)
        time = [
            (start_time + datas_index * 600).datetime
            for datas_index in range(datas.shape[1])
        ]
        df = pd.DataFrame(zip(*datas), columns=columns, index=pd.Series(time))

        save_dataframe(df, csv_path, index=True, index_label="time")
        return None

    def _check_transform(self, name) -> bool:
        if name not in self.df.columns and name in self.parent.data_streams:
            return True
        else:
            return False

    def compute_transforms(self):
        for col in self.df.columns:
            if col is "time":
                continue

            # inverse
            if self._check_transform("inv_" + col):
                self.df["inv_" + col] = 1.0 / self.df[col]

            # diff
            if self._check_transform("diff_" + col):
                self.df["diff_" + col] = self.df[col].diff()
                self.df["diff_" + col][0] = 0.0

            # log
            if self._check_transform("log_" + col):
                self.df["log_" + col] = np.log10(self.df[col])

            # stft
            if self._check_transform("stft_" + col):
                seg, freq = [12, 16]
                data = pd.Series(np.zeros(seg * 6 - 1))
                data = data.append(self.df[col], ignore_index=True)
                z = abs(
                    stft(
                        data.values,
                        window="nuttall",
                        nperseg=seg * 6,
                        noverlap=seg * 6 - 1,
                        boundary=None,
                    )[2]
                )
                self.df["stft_" + col] = np.mean(z[freq : freq + 2, :], axis=0)

            if self._check_transform("zsc_" + col):
                # log data
                dt = np.log10(self.df[col]).replace([np.inf, -np.inf], np.nan).dropna()

                # Drop test data
                if len(self.parent.exclude_dates) != 0:
                    for exclude_date_range in self.parent.exclude_dates:
                        t0, t1 = [to_datetime(date) for date in exclude_date_range]
                        inds = (dt.index < t0) | (dt.index >= t1)
                        dt = dt.loc[inds]

                # Record mean/std/min
                mn = np.mean(dt)
                std = np.std(dt)
                min_z_score = np.min(dt)

                # Calculate percentile
                self.df["zsc_" + col] = (np.log10(self.df[col]) - mn) / std
                # self.df['zsc_'+col]=(self.df[col]-mn)/std
                self.df["zsc_" + col] = self.df["zsc_" + col].fillna(min_z_score)
                self.df["zsc_" + col] = 10 ** self.df["zsc_" + col]
            if self._check_transform("zsc2_" + col):
                # log data
                dt = np.log10(self.df[col]).replace([np.inf, -np.inf], np.nan).dropna()

                # Drop test data
                if len(self.parent.exclude_dates) != 0:
                    for exclude_date_range in self.parent.exclude_dates:
                        t0, t1 = [to_datetime(date) for date in exclude_date_range]
                        inds = (dt.index < t0) | (dt.index >= t1)
                        dt = dt.loc[inds]

                # Record mean/std/min
                mn = np.mean(dt)
                std = np.std(dt)
                min_z_score = np.min(dt)

                # Calculate percentile
                self.df["zsc2_" + col] = (np.log10(self.df[col]) - mn) / std
                # self.df['zsc_'+col]=(self.df[col]-mn)/std
                self.df["zsc2_" + col] = self.df["zsc2_" + col].fillna(min_z_score)
                self.df["zsc2_" + col] = 10 ** self.df["zsc2_" + col]

                self.df["zsc2_" + col] = self.df["zsc2_" + col].rolling(window=2).min()
                self.df["zsc2_" + col][0] = self.df["zsc2_" + col][1]
            if self._check_transform("log_zsc2_" + col):
                # log data
                dt = np.log10(self.df[col]).replace([np.inf, -np.inf], np.nan).dropna()

                # Drop test data
                if len(self.parent.exclude_dates) != 0:
                    for exclude_date_range in self.parent.exclude_dates:
                        t0, t1 = [to_datetime(date) for date in exclude_date_range]
                        inds = (dt.index < t0) | (dt.index >= t1)
                        dt = dt.loc[inds]

                # Record mean/std/min
                mn = np.mean(dt)
                std = np.std(dt)
                min_z_score = np.min(dt)

                # Calculate percentile
                self.df["log_zsc2_" + col] = (np.log10(self.df[col]) - mn) / std

                # self.df['zsc_'+col]=(self.df[col]-mn)/std
                self.df["log_zsc2_" + col] = self.df["log_zsc2_" + col].fillna(
                    min_z_score
                )
                self.df["log_zsc2_" + col] = 10 ** self.df["log_zsc2_" + col]
                self.df["log_zsc2_" + col] = (
                    self.df["log_zsc2_" + col].rolling(window=2).min()
                )
                self.df["log_zsc2_" + col] = np.log10(self.df["log_zsc2_" + col])
                self.df["log_zsc2_" + col][0] = self.df["log_zsc2_" + col][1]
            if self._check_transform("diff_zsc2_" + col):

                # log data
                dt = np.log10(self.df[col]).replace([np.inf, -np.inf], np.nan).dropna()

                # Drop test data
                if len(self.parent.exclude_dates) != 0:
                    for exclude_date_range in self.parent.exclude_dates:
                        t0, t1 = [to_datetime(date) for date in exclude_date_range]
                        inds = (dt.index < t0) | (dt.index >= t1)
                        dt = dt.loc[inds]

                # Record mean/std/min
                mn = np.mean(dt)
                std = np.std(dt)
                min_z_score = np.min(dt)

                # Calculate percentile
                self.df["diff_zsc2_" + col] = (np.log10(self.df[col]) - mn) / std
                # self.df['zsc_'+col]=(self.df[col]-mn)/std
                self.df["diff_zsc2_" + col] = self.df["diff_zsc2_" + col].fillna(
                    min_z_score
                )
                self.df["diff_zsc2_" + col] = 10 ** self.df["diff_zsc2_" + col]
                self.df["diff_zsc2_" + col] = (
                    self.df["diff_zsc2_" + col].rolling(window=2).min()
                )
                self.df["diff_zsc2_" + col] = self.df[col].diff()
                self.df["diff_zsc2_" + col][0] = 0.0

    def get_data(self, datetime_start=None, datetime_end=None) -> pd.DataFrame:
        """Return tremor data in requested date range.
        Parameters:
        -----------
        ti : str, datetime.datetime
            Date of first data point (default is earliest data).
        tf : str, datetime.datetime
            Date of final data point (default is latest data).
        Returns:
        --------
        df : pandas.DataFrame
            Data object truncated to requested date range.
        """
        # set date range defaults
        if datetime_start is None:
            datetime_start = self.datetime_start
        if datetime_end is None:
            datetime_end = self.datetime_end

        # convert datetime format
        datetime_start = to_datetime(datetime_start)
        datetime_end = to_datetime(datetime_end)

        # subset data
        indices = (self.df.index >= datetime_start) & (self.df.index < datetime_end)
        return self.df.loc[indices]

    def is_eruption_in(self, days, from_time):
        """Binary classification of eruption imminence.
        Parameters:
        -----------
        days : float
            Length of look-forward.
        from_time : datetime.datetime
            Beginning of look-forward period.
        Returns:
        --------
        label : int
            1 if eruption occurs in look-forward, 0 otherwise

        """
        for te in self.tes:
            if 0 < (te - from_time).total_seconds() / (3600 * 24) < days:
                return 1.0
        return 0.0

    def _validate(self):
        """
        Load an existing file and check the date range of data.
        """
        with open(self.eruptive_file, "r") as fp:
            self.tes = [to_datetime(_line.rstrip()) for _line in fp.readlines()]

        # Check if a tremor data file exists. If not, create one
        if not self.tremor_file_exists:
            if self.verbose:
                print(f"Creating new tremor data...")
            df = pd.DataFrame(columns=self.cols)
            df.to_csv(self.tremor_file, index_label="time")
            if self.verbose:
                print(f"New tremor data saved to: {self.tremor_file}")

        # Load dataframe
        self.df = load_dataframe(
            self.tremor_file, index_col=0, parse_dates=True, infer_datetime_format=True
        )

        if len(self.df) > 0:
            self.datetime_start = self.df.index[0]
            self.datetime_end = self.df.index[-1]
            if self.verbose:
                print(f"Dataframe loaded: {self.df.shape}")
                print(f"Start date: {self.datetime_start}")
                print(f"End date: {self.datetime_end}")

            return None

        print(f"⚠️ Data not found in tremor file :: {self.tremor_file}")
        return None
