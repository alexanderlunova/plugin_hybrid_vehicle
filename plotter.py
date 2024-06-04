import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np

def plot_timeseries(start_time, end_time, time_step, data_dict):
    data_dict = data_dict
    # Create timestamps for 24 hours
    start_time = pd.Timestamp(start_time)  # Start date of your data
    end_time = start_time + pd.Timedelta(hours=end_time)  # End date of your data
    time_index = pd.date_range(start=start_time, end=end_time, freq='15T')

    plot_keys = ["distance_ts", "SOC_ts", "el_consumption_ts", "ice_consumption_ts"]
    data_dict = {key: data_dict[key] for key in plot_keys if key in data_dict}

    units = {
        "distance_ts": "Distance [km]",
        "SOC_ts": "SOC [%]",
        "el_consumption_ts": "Consumption [kWh]",
        "ice_consumption_ts": "Consumption [Liters]"
    }

    # Titles
    titles = {
        "distance_ts": "Driven distance",
        "SOC_ts": "SOC of Car Battery",
        "el_consumption_ts": "Energy Consumption",
        "ice_consumption_ts": "Fuel Consumption"
    }

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Plot each time series on its corresponding subplot
    for idx, (column, data) in enumerate(data_dict.items()):
        row = idx // 2
        col = idx % 2
        ax = axs[row, col]

        # Create a DataFrame
        data_df = pd.DataFrame({'Timestamp': time_index[:len(data)], 'Value': data})
        ax.plot(data_df['Timestamp'], data_df['Value'], marker='o', linestyle='-',
                markersize=0, linewidth=1)  # Adjust markersize here

        ax.set_ylabel(units.get(column))
        ax.set_title(titles.get(column))
        ax.xaxis.set_major_formatter(DateFormatter("%d-%m %H:%M"))
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def plot_soc_frequency(soc_ts, distance_ts, settings):
    mask = distance_ts != 0
    soc_lower_limit = settings["SOC_discharge_limit"]
    soc_upper_limit = settings["charging_limit"]
    plt.hist(soc_ts[mask], bins=50, weights=np.ones(len(soc_ts[mask])) / len(soc_ts[mask]))
    plt.axvline(soc_lower_limit, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(soc_upper_limit, color='g', linestyle='dashed', linewidth=1)
    plt.show()
