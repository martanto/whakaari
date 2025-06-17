from __future__ import annotations

import os
from typing import List
from .tremor_data import TremorData
from .utils import to_datetime
from datetime import timedelta


class ForecastModel:
    def __init__(
        self,
        station: str,
        start_date: str,
        end_date: str,
        window: float,
        overlap: float,
        look_forward: float,
        eruptive_file: str,
        tremor_data_file: str = None,
        data_streams: List[str] = None,
        exclude_dates: List[str] = None,
        root: str = None,
        feature_root: str = None,
        feature_dir: str = None,
        n_jobs: int = 2,
        savefile_type="pkl",
        verbose: bool = False,
    ):
        if data_streams is None:
            data_streams = ["rsam", "mf", "hf", "dsar"]

        self.verbose = verbose
        self.station = station
        self.data_streams = data_streams
        self.exclude_dates = exclude_dates

        # Length of look-forward in days
        self.look_forward: float = look_forward
        self.savefile_type = savefile_type

        self.data: TremorData = TremorData(
            station=station,
            parent=self,
            data_dir=os.getcwd(),
            eruptive_file=eruptive_file,
            tremor_data_file=tremor_data_file,
        )

        if any([column not in self.data.df.columns for column in data_streams]):
            raise ValueError(f"❌ Data restricted to any of {self.data.df.columns}")

        if any(["_" in column for column in data_streams]):
            self.data.compute_transforms()

        if start_date is None:
            start_date = self.data.datetime_start
        self.start_date = start_date

        if end_date is None:
            end_date = self.data.datetime_end
        self.end_date = end_date

        # Originally self.ti_model and self.tf_model
        self.start_date_model = to_datetime(start_date)
        self.end_date_model = to_datetime(end_date)

        if self.end_date_model > self.data.datetime_end:
            t0, t1 = [
                self.end_date_model.strftime("%Y-%m-%d %H:%M"),
                self.data.datetime_end.strftime("%Y-%m-%d %H:%M"),
            ]
            raise ValueError(
                "Model end date '{:s}' beyond data range '{:s}'".format(t0, t1)
            )
        if self.start_date_model < self.data.datetime_start:
            t0, t1 = [
                self.start_date_model.strftime("%Y-%m-%d %H:%M"),
                self.data.datetime_start.strftime("%Y-%m-%d %H:%M"),
            ]
            raise ValueError(
                "Model start date '{:s}' predates data range '{:s}'".format(t0, t1)
            )

        # Length of look-forward.
        self.dtf = timedelta(days=look_forward)

        # Length between data samples (10 minutes).
        self.dt = timedelta(seconds=600)

        # Number of samples in window.
        self.iw = int(window * 6 * 24)

        # Number of samples in overlapping section of window.
        self.io = int(overlap * self.iw)
        if self.io == self.iw:
            self.io -= 1

        # Length of data window in days.
        self.window = self.iw * 1.0 / (6 * 24)

        # Length of window.
        self.dtw = timedelta(days=self.window)

        if self.start_date_model - self.dtw < self.data.datetime_start:
            self.start_date_model = self.data.datetime_start + self.dtw

        # Fraction of overlap between adjacent windows. Set this to 1.
        # For overlap of entire window minus 1 data point.
        self.overlap = self.io * 1.0 / self.iw

        # Length of non-overlapping section of window
        self.dto = (1.0 - self.overlap) * self.dtw

        self.drop_features = []
        self.exclude_dates = []
        self.use_only_features = []
        self.compute_only_features = []
        self.update_feature_matrix = True
        self.n_jobs = n_jobs

        # naming convention and file system attributes
        if root is None:
            root = (
                f"fm_{self.window:3.2f}_{self.overlap:3.2f}_{self.look_forward:3.2f}_"
            )
            root += (("{:s}-" * len(self.data_streams))[:-1]).format(
                *sorted(self.data_streams)
            )
        self.root = root

        self.feature_root = feature_root
        self.root_dir = os.getcwd()

        self.output_dir = os.path.join(self.root_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.plot_dir = os.path.join(self.output_dir, "plots", self.root)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.model_dir = os.path.join(self.output_dir, "models", self.root)
        os.makedirs(self.model_dir, exist_ok=True)

        if feature_dir is None:
            feature_dir = os.path.join(self.output_dir, "features")
        self.feature_dir = feature_dir
        os.makedirs(self.feature_dir, exist_ok=True)

        self.feature_file = lambda feature_name, ds: os.path.join(
            feature_dir,
            f"fm_{window:3.2f}w_{ds}_{station}{feature_name}.{savefile_type}",
        )

        self.prediction_dir = os.path.join(self.output_dir, "predictions", self.root)
        os.makedirs(self.prediction_dir, exist_ok=True)

        if verbose:
            print(f"Start date: {self.start_date_model.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End date: {self.end_date_model.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Tremor data: {self.data.tremor_file}")
            print(f"Root dir: {self.root_dir}")
            print(f"Output dir: {self.output_dir}")
            print(f"Plot dir: {self.plot_dir}")
            print(f"Model dir: {self.model_dir}")
            print(f"Feature dir: {self.feature_dir}")
            print(f"Prediction dir: {self.prediction_dir}")
