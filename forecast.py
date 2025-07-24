#%%
from whakaari import ForecastModel
import warnings

warnings.filterwarnings("ignore", module="tsfresh")

#%%
def main():
    data_streams = ["rsam", "mf", "hf", "dsar"]
    start_date = "2025-01-01"
    end_date = "2025-05-26"

    fm = ForecastModel(
        station="OJN",
        start_date=start_date,
        end_date=end_date,
        window=2.0,
        overlap=0.75,
        look_forward=2.0,
        eruptive_file=r"D:\Projects\whakaari\input\OJN_eruptive_periods.txt",
        tremor_data_file=r"D:\Projects\whakaari\output\OJN_tremor_data.csv",
        data_streams=data_streams,
        verbose=True,
    )

    drop_features = ["linear_trend_timewise", "agg_linear_trend"]

    fm.train(
        start_date=start_date,
        end_date=end_date,
        drop_features=drop_features,
        retrain_model=True,
        n_jobs=4,
    )


#%%
if __name__ == "__main__":
    main()
