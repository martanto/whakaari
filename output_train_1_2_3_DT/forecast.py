#%%
from whakaari import ForecastModel
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore", module="tsfresh")

#%%
def main():
    data_streams = ["rsam", "mf", "hf", "dsar"]
    start_date = "2025-01-01"
    # end_date = "2025-05-26"
    end_date = "2025-05-31"
    day = timedelta(days=1)

    fm = ForecastModel(
        station="OJN",
        start_date=start_date,
        end_date=end_date,
        window=2.0,
        overlap=0.75,
        look_forward=2.0,
        eruptive_file=r"D:\Project\whakaari\input\OJN_eruptive_periods.txt",
        tremor_data_file=r"D:\Project\whakaari\output\OJN_tremor_data.csv",
        data_streams=data_streams,
        verbose=True,
    )

    drop_features = ["linear_trend_timewise", "agg_linear_trend"]

    fm.train(
        start_date=start_date,
        end_date=end_date,
        drop_features=drop_features,
        retrain_model=False,
        classifier="DT",
        n_jobs=16,
    )

    fm.hires_forecast(
        start_date="2025-06-10",
        end_date="2025-06-20",
        recalculate=False,
        save='OJN_forecast_1_and_2_DT_0.7_2025-06-10_2025-06-20.png',
        nz_timezone=False,
        threshold = 0.7,
        n_jobs=16
    )


#%%
if __name__ == "__main__":
    main()
