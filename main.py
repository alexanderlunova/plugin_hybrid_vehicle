import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PHEV import Car
from matplotlib.dates import DateFormatter
def read_csv_data(file_path):
    dtype = {
        'ID': str,
        'Distance_km': float,
        'Consumption_kWh': float,
        'PowerRating_kW': float,
        'Load_kW': float,
        'SoC': float,
        'Load_kW.1': float,
        'SoC.1': float,
        'Load_kW.2': float,
        'SoC.2': float,
        'Load_kW.3': float,
        'SoC.3': float
    }

    # Define a date parser function matching your date format
    date_parser = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

    # Read the CSV file with specified dtypes and date parser
    df = pd.read_csv(file_path, dtype=dtype, low_memory=False, skiprows=1)

    # Create a dictionary to hold DataFrames for each year
    driving_profiles = {}

    # Group by year and create a DataFrame for each year
    for id, group in df.groupby('ID'):
        driving_profiles[id] = group.reset_index(drop=True)

    return driving_profiles

def plot_timeseries(start_time, end_time, time_step, data_dict):
    # Create timestamps for 24 hours
    start_time = pd.Timestamp(start_time)  # Start date of your data
    end_time = start_time + pd.Timedelta(hours=end_time)  # End date of your data
    time_index = pd.date_range(start=start_time, end=end_time, freq='15T')

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
                markersize=3)  # Adjust markersize here
        ax.set_ylabel(units.get(column))
        ax.set_title(titles.get(column))
        ax.xaxis.set_major_formatter(DateFormatter("%d-%m %H:%M"))
        ax.grid(True)
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()


def run_simulation(configuration):

    t = configuration["start_time"]
    T = configuration["end_time"]
    delta_t = configuration["delta_t"]

    simulation_results = {
        "distance_ts": [],
        "SOC_ts": [],
        "el_consumption_ts": [],
        "ice_consumption_ts": []

    }

    car = configuration["car"]
    ts_consumption = configuration["ts_consumption"]
    ts_distance = configuration["ts_distance"]


    while t < T:
        el_consumption, ice_consumption = (
            car.calc_consumption(ts_consumption[t*1/delta_t], ts_distance[t*1/delta_t], delta_t))
        car.drive(el_consumption, ice_consumption, ts_distance[t*1/delta_t], delta_t, t)
        t = t + delta_t

        simulation_results["distance_ts"].append(ts_distance[t*1/delta_t])
        simulation_results["SOC_ts"].append(car.SOC)

    simulation_results["el_consumption_ts"] = car.el_consumption_ts
    simulation_results["ice_consumption_ts"] = car.ice_consumption_ts

    return simulation_results

def find_special_days(df):

    num_bins = 365
    days = []
    for i in range(num_bins):
        start = i * 24
        end = (i + 1) * 24
        bin_data = sum(df['Distance_km'][start:end])
        if(bin_data>300):
            days.append(i)


    return days


def setup(settings):

    driving_profiles = read_csv_data(settings["driving_data"])
    df = driving_profiles[settings["ID"]]
    ts_distance = df['Distance_km']
    ts_consumption = df['Consumption_kWh']

    car = Car(
            battery_size=settings["battery_size"],
            fuel_tank_size=settings["fuel_tank_size"],
            SOC=settings["SOC"],
            ice_consumption_per_100km=settings["ice_consumption_per_100km"],
            SOC_limit=settings["SOC_limit"],
            max_el_power=settings["max_el_power"],
            max_ice_power=settings["max_ice_power"],
            tank_level=settings["tank_level"],
            recharge=settings["recharge"],
            charging_limit=settings["charging_limit"]
    )

    configuration = {
        "ts_distance": ts_distance,
        "ts_consumption": ts_consumption,
        "car": car,
        "delta_t": settings["delta_t"],
        "end_time": settings["end_time"],
        "start_time": settings["start_time"]
    }

    return configuration

def main():

    settings = {
        "ID": "1.0",
        "ice_consumption_per_100km": 5,
        "battery_size": 0.001,
        "fuel_tank_size": 50,
        "SOC": 0.9,
        "charging_limit": 0.9,
        "SOC_limit": 0.2,
        "max_el_power": 100,
        "max_ice_power": 100,
        "tank_level": 30,
        "driving_data":
            r"C:\Users\alexl\Downloads\emobpy_timeseries_original\emobpy_timeseries_original.csv",
        "delta_t": 0.25,
        "end_time": 24*120,
        "start_time": 24*100,
        "plot": True,
        "recharge": False
    }

    configuration = setup(settings)

    simulation_results = run_simulation(configuration)

    if(settings["plot"]):
        plot_timeseries(configuration["start_time"], configuration["end_time"],
                        configuration["delta_t"], simulation_results)

    return 0




if __name__ == "__main__":
    main()



