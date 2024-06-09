import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PHEV import Car
from matplotlib.dates import DateFormatter


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



    # calculate calendar ageing for whole simulation time
    calendar_capacity_fade = calculate_calendar_ageing(settings["simulation_years"])

    configuration = {
        "ts_distance": ts_distance,
        "ts_consumption": ts_consumption,
        "delta_t": settings["delta_t"],
        "end_time": settings["end_time"],
        "start_time": settings["start_time"],
        "cost_parameters": settings["cost_parameters"],
        "simulation_years": settings["simulation_years"],
        "calendar_capacity_fade": calendar_capacity_fade,
        "battery_size": settings["battery_size"],
        "additional_battery_weight": settings["additional_battery_weight"]
    }

    return configuration


def setup_car(settings):

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

    return car



def calculate_calendar_ageing(simulation_years):

    soc = 50  # SOC in %
    T = 30  # Temperature in C°
    t = simulation_years * 12

    # fitted curve for calendar aging
    calendar_capacity_fade = np.zeros(t)
    for t in range(t):
        calendar_capacity_fade[t] = 0.0025 * pow(np.e, 0.1099 * T) * pow(np.e, 0.0169 * soc) * \
                                    pow(t, (-3.866 * pow(10, -13)) * pow(T, 6.635) +
                                        (-4.853 * pow(10, -12)) * pow(soc, 5.508) + 0.9595) + 0.7

    return calendar_capacity_fade


def calculate_cycle_ageing(dod_ts):

    t = 12
    # cycle lifetime
    doc = 0.8  # cycle depth of used data for fitted curve
    c_rate = 4  # conservative c_rate due to too low c_rates for fitted curve
    fec = sum(dod_ts)/2

    k_crate = 0.0630 * c_rate + 0.0971  # c-rate dependent factor
    k_doc = 4.02 * pow(doc - 0.6, 3) + 1.0923  # dod dependent factor
    k_T = 1  # temperature dependent factor(1 for ambient temperatures)

    cycle_capacity_fade = np.zeros(t)
    for i, t in enumerate(range(t)):
        k_fec = pow(fec * t / 12, 0.5)  # full equivalent cycle dependent factor
        cycle_capacity_fade[i] = k_crate * k_doc * k_fec * k_T

    return cycle_capacity_fade


def calculate_final_costs(configuration, ice_consumption, charging_consumption, cost_parameters):

    discount_rate = 0.05
    years = configuration["simulation_years"]

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

    # Calculate the annuity factor
    annuity_factor = (pow(1 + discount_rate, years) - 1) / (discount_rate * pow(1 + discount_rate, years))

    # Calculate NPV of OPEX
    npv_opex = running_costs * annuity_factor / years

    # NPV of CAPEX is typically the fixed costs as they are considered upfront
    npv_capex = fixed_costs

    # Calculate the total NPV
    npv = npv_opex + npv_capex

    # Calculate the yearly annuity
    yearly_annuity = npv * (discount_rate * pow(1 + discount_rate, years)) / (pow(1 + discount_rate, years) - 1)

    return {
        "npv_opex": npv_opex,
        "npv_capex": npv_capex,
        "npv": npv,
        "yearly_annuity": yearly_annuity,
        "running_costs": running_costs,
        "fixed_costs": fixed_costs,
        "electricity_costs": electricity_costs,
        "fuel_costs": fuel_costs,
        "maintenance_costs": maintenance_costs
    }


def print_cost_breakdown(configuration, ice_consumption, charging_consumption, cost_parameters):
    discount_rate = 0.05
    years = configuration["simulation_years"]

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

    factors = []
    for t in range(years):
        factors.append(pow(1-discount_rate, t -1))

    annuity_factor = np.sum(factors)

    npv_opex = running_costs * annuity_factor
    npv_capex = fixed_costs
    npv = npv_opex + npv_capex

    yearly_annuity = npv * (discount_rate * pow(1 + discount_rate, years)) / ((pow(1 + discount_rate, years)) - 1)

    print(f"{'Cost_breakdown':<30}{'Euro':>10}")
    print(f"{'-' * 40}")
    print(f"{'Running Costs':<30}{running_costs:>10.2f}")
    print(f"{'Electricity Costs':<30}{cost_parameters["electricity_costs"] * sum(charging_consumption):>10.2f}")
    print(f"{'Fuel Costs':<30}{cost_parameters["fuel_costs"] * sum(ice_consumption):>10.2f}")
    print(f"{'NPV OPEX':<30}{npv_opex:>10.2f}")
    print(f"{'Upfront costs':<30}{npv_capex:>10.2f}")
    print(f"{'NPV':<30}{npv:>10.2f}")
    print(f"{'Yearly annuity':<30}{yearly_annuity:>10.2f}")
    print(f"{'-' * 40}")

    costs = {
        "running_costs": running_costs,
        "electricity_costs": electricity_costs,
        "fuel_costs": fuel_costs,
        "maintenance_costs": maintenance_costs,
    }

    return costs


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


def print_stats(simulation_results, configuration):
    print(f"{'Driving_data':<30}{'Value':>10}")
    print(f"{'-' * 40}")
    print(f"{'Total distance driven':<30}{sum(simulation_results["distance_ts"]):>10.2f}")
    print(f"{'Total el consumption':<30}{sum(simulation_results["el_consumption_ts"]):>10.2f}")
    print(f"{'Total ice consumption':<30}{sum(simulation_results["ice_consumption_ts"]):>10.2f}")
    print(f"{'Total charging consumption':<30}{(sum(simulation_results["charging_consumption_ts"])):>10.2f}")
    print(f"{'-' * 40}")


def modify_driving_profiles(configuration):

    num_bins = 365
    long_days = []
    short_days = []
    new_ts_distance = []

    for i in range(num_bins):
        start = i * 96
        end = (i + 1) * 96
        bin_data = sum(configuration["ts_distance"][start:end])
        if (bin_data > 400):
            long_days.append([start,end])
            print(bin_data)
        elif(bin_data < 42 and bin_data > 38):
            short_days.append([start,end])

    long_distances = np.tile(np.array(configuration["ts_distance"][long_days[1][0]:long_days[1][1]]), 10)
    long_consumption = np.tile(np.array(configuration["ts_consumption"][long_days[1][0]:long_days[1][1]]), 10)

    short_distances = np.tile(np.array(configuration["ts_distance"][short_days[1][0]:short_days[1][1]]), 355)
    short_consumption = np.tile(np.array(configuration["ts_consumption"][short_days[1][0]:short_days[1][1]]), 355)

    distance_values = np.concatenate((long_distances,short_distances))
    consumption_values = np.concatenate((long_consumption, short_consumption))

    # Combine the lists into a list of tuples
    combined_lists = list(zip(distance_values, consumption_values))

    # Shuffle the combined list of tuples
    #np.random.shuffle(combined_lists)

    # Unzip the shuffled list of tuples back into separate lists
    distance_values, consumption_values = zip(*combined_lists)

    distance_values = list(distance_values)
    consumption_values = list(consumption_values)

    configuration["ts_distance"] = pd.Series(distance_values, name="ts_distance")
    configuration["ts_consumption"] = pd.Series(consumption_values, name="ts_consumption")

    return configuration
