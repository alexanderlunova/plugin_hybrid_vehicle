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


def run_simulation(configuration, simulation_year):
    start_time = configuration["start_time"]
    end_time = configuration["end_time"]
    delta_t = configuration["delta_t"]

    timeframe_start = int((simulation_year - 1) * 8760 * 1 / configuration["delta_t"])
    timeframe_end = int(simulation_year * 8760 * 1 / configuration["delta_t"])

    simulation_results = {}

    car = configuration["car"]
    ts_consumption = configuration["ts_consumption"]
    ts_distance = configuration["ts_distance"]

    time_step = start_time
    while time_step < end_time:
        # Overall simulation time
        time = int((time_step + (simulation_year - 1) * 8760) * 1 / delta_t)

        # Calculate consumption with driving profiles
        car.calc_consumption(ts_consumption[time_step * 1 / delta_t], ts_distance[time_step * 1 / delta_t], delta_t,
                             time)

        # Simulate driving with new battery, fuel consumption
        car.drive(delta_t, time)

        time_step = time_step + delta_t

    simulation_results["SOC_ts"] = car.SOC_ts[timeframe_start:timeframe_end]
    simulation_results["distance_ts"] = ts_distance
    simulation_results["el_consumption_ts"] = car.el_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["ice_consumption_ts"] = car.ice_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["charging_consumption_ts"] = car.charging_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["battery_capacity"] = car.capacity_ts[timeframe_start:timeframe_end]

    return simulation_results


def find_special_days(df):
    num_bins = 365
    days = []
    for i in range(num_bins):
        start = i * 96
        end = (i + 1) * 96
        bin_data = sum(df[start:end])
        if (bin_data > 300):
            days.append(i)

    return days


def setup(settings):
    driving_profiles = read_csv_data(settings["driving_data"])
    df = driving_profiles[settings["ID"]]
    ts_distance = df['Distance_km']
    ts_consumption = df['Consumption_kWh']

    print(find_special_days(df['Distance_km']))

    car = Car(
        battery_size=settings["battery_size"],
        fuel_tank_size=settings["fuel_tank_size"],
        soc=settings["SOC"],
        ice_consumption_per_100km=settings["ice_consumption_per_100km"],
        soc_discharge_limit=settings["SOC_discharge_limit"],
        max_el_power=settings["max_el_power"],
        max_ice_power=settings["max_ice_power"],
        tank_level=settings["tank_level"],
        recharge=settings["recharge"],
        charging_limit=settings["charging_limit"],
        charging_interval=settings["charging_interval"],
        c_rate=settings["c_rate"],
        el_charging_efficiency=settings["el_charging_efficiency"],
        el_discharging_efficiency=settings["el_discharging_efficiency"]
    )

    configuration = {
        "ts_distance": ts_distance,
        "ts_consumption": ts_consumption,
        "car": car,
        "delta_t": settings["delta_t"],
        "end_time": settings["end_time"],
        "start_time": settings["start_time"],
        "cost_parameters": settings["cost_parameters"],
        "simulation_years": settings["simulation_years"]
    }

    return configuration


def calculate_costs(configuration, ice_consumption, charging_consumption, cost_parameters):
    discount_rate = 0.01
    years = 1

    electricity_costs = cost_parameters["electricity_costs"] * sum(charging_consumption)
    fuel_costs = cost_parameters["fuel_costs"] * sum(ice_consumption)
    maintenance_costs = cost_parameters["maintenance_costs"] * sum(configuration["ts_distance"])

    running_costs = (
            fuel_costs + electricity_costs + maintenance_costs
    )

    fixed_costs = (
            cost_parameters["vehicle_costs"] - cost_parameters["build_in_battery_costs"] +
            cost_parameters["cell_amount"] * cost_parameters["battery_cell_price"]
    )

    annuity_factor = (1 - (1 + discount_rate) ** -years) / discount_rate
    present_value_running_costs = running_costs * annuity_factor
    total_present_value = fixed_costs + present_value_running_costs

    print(f"{'Cost_breakdown':<30}{'Euro':>10}")
    print(f"{'-' * 40}")
    print(f"{'Running Costs':<30}{running_costs:>10.2f}")
    print(f"{'Electricity Costs':<30}{cost_parameters["electricity_costs"] * sum(charging_consumption):>10.2f}")
    print(f"{'Fuel Costs':<30}{cost_parameters["fuel_costs"] * sum(ice_consumption):>10.2f}")
    print(f"{'Present value running costs':<30}{present_value_running_costs:>10.2f}")
    print(f"{'Total present value':<30}{total_present_value:>10.2f}")
    print(f"{'-' * 40}")

    costs = {
        "running_costs": running_costs,
        "electricity_costs": electricity_costs,
        "fuel_costs": fuel_costs,
        "maintenance_costs": maintenance_costs
    }

    return costs


def print_stats(simulation_results, configuration):
    print(f"{'Driving_data':<30}{'Value':>10}")
    print(f"{'-' * 40}")
    print(f"{'Total distance driven':<30}{sum(simulation_results["distance_ts"]):>10.2f}")
    print(f"{'Total el consumption':<30}{sum(simulation_results["el_consumption_ts"]):>10.2f}")
    print(f"{'Total ice consumption':<30}{sum(simulation_results["ice_consumption_ts"]):>10.2f}")
    print(f"{'Total charging consumption':<30}{(sum(simulation_results["charging_consumption_ts"])):>10.2f}")
    print(f"{'-' * 40}")


def experiment_1():
    settings = {
        "ID": "1.0",
        "simulation_years": 10,
        "ice_consumption_per_100km": 10,
        "battery_size": 20,
        "fuel_tank_size": 50,
        "SOC": 0.9,  # starting SOC
        "el_charging_efficiency": 0.98,
        "el_discharging_efficiency": 0.9,
        "charging_limit": 0.9,
        "charging_interval": [93, 94, 95, 96, 0, 1, 2, 3, 4],
        "c_rate": 0.5,
        "SOC_discharge_limit": 0.2,
        "max_el_power": 100,
        "max_ice_power": 100,
        "tank_level": 30,
        "car_weight": 2000,
        "battery_weight": 800,
        "driving_data":
            r"C:\Users\alexl\Downloads\emobpy_timeseries_original\emobpy_timeseries_original.csv",
        "delta_t": 0.25,
        "end_time": 24 * 365 - 1,
        "start_time": 24 * 0,
        "years": 1,
        "plot": False,
        "recharge": True,
        "calc_econ": True,
        "cost_parameters": {
            "vehicle_costs": 30000,  # Costs in Euro
            "build_in_battery_costs": 8000,
            "battery_cell_price": 0.5,  # Costs in Euro/Cell
            "cell_amount": 15000,
            "fuel_costs": 1.6,  # Costs in Euro/Liter
            "electricity_costs": 0.4,  # Costs in Euro/kWh
            "battery_aging_factor": 1.5,
            "maintenance_costs": 0.0096  # Maintenance costs in euro/km
        }
    }

    experiment_results = {
        "battery_capacity": [],
        "total_present_value": [],
        "fuel_consumption": [],
        "charging_consumption": [],
        "running_costs": [],
        "electricity_costs": [],
        "fuel_costs": [],
        "maintenance_costs": []
    }

    configuration = setup(settings)

    t = configuration["simulation_years"]
    for year in range(1, t + 1):

        simulation_results = run_simulation(configuration, year)

        print_stats(simulation_results, configuration)
        costs = calculate_costs(configuration,
                                simulation_results["ice_consumption_ts"],
                                simulation_results["charging_consumption_ts"],
                                configuration["cost_parameters"])

        if settings["plot"]:
            plot_timeseries(configuration["start_time"], configuration["end_time"],
                            configuration["delta_t"], simulation_results)

        experiment_results["running_costs"].append(costs["running_costs"])
        experiment_results["fuel_costs"].append(costs["fuel_costs"])
        experiment_results["electricity_costs"].append(costs["electricity_costs"])
        experiment_results["maintenance_costs"].append(costs["maintenance_costs"])

        experiment_results["battery_capacity"].append(np.mean(simulation_results["battery_capacity"]))
        experiment_results["fuel_consumption"].append(sum(simulation_results["ice_consumption_ts"]))
        experiment_results["charging_consumption"].append(sum(simulation_results["charging_consumption_ts"]))

    years = range(1, t + 1)
    # Create a 2x2 multiplot
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Cost plot for yearly Total present value
    axs[0, 0].plot(years, experiment_results["running_costs"], label='El + Fuel Costs', marker='o')
    axs[0, 0].plot(years, experiment_results["electricity_costs"], marker='o', label='Electricity Costs', color='green')
    axs[0, 0].plot(years, experiment_results["fuel_costs"], marker='o', label='Fuel Costs', color='orange')
    axs[0, 0].plot(years, experiment_results["maintenance_costs"], marker='o', label='Maintenance Costs', color='grey')
    axs[0, 0].set_title('Yearly Running Costs')
    axs[0, 0].set_xlabel('Year')
    axs[0, 0].set_ylabel('Running costs (Euro)')
    axs[0, 0].grid(True)
    axs[0, 0].legend(loc='upper left')

    # Plot 2: Plot for battery capacity over years
    axs[0, 1].plot(years, experiment_results["battery_capacity"], marker='o')
    axs[0, 1].set_title('Battery Capacity Over Years')
    axs[0, 1].set_xlabel('Year')
    axs[0, 1].set_ylabel('Battery Capacity (kWh)')
    axs[0, 1].grid(True)

    # Plot 3: Plot for yearly fuel consumption
    axs[1, 0].plot(years, experiment_results["fuel_consumption"], marker='o')
    axs[1, 0].set_title('Yearly Fuel Consumption')
    axs[1, 0].set_xlabel('Year')
    axs[1, 0].set_ylabel('Fuel Consumption (liters)')
    axs[1, 0].grid(True)

    # Plot 4: Plot for yearly electricity consumption
    axs[1, 1].plot(years, experiment_results["charging_consumption"], marker='o')
    axs[1, 1].set_title('Yearly Electricity Consumption')
    axs[1, 1].set_xlabel('Year')
    axs[1, 1].set_ylabel('Electricity Consumption (kWh)')
    axs[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    experiment_1()

    return 0


if __name__ == "__main__":
    main()
