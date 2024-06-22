import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PHEV import Car
from matplotlib.dates import DateFormatter
from helper_functions import *
from plotter import *
import random


def run_simulation(configuration, simulation_year=1):
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
                             time, configuration["additional_battery_weight"])

        # Simulate driving with new battery, fuel consumption
        car.drive(delta_t, time)

        time_step = time_step + delta_t

    # Calculate calendar ageing capacity fade
    car.battery_size = (configuration["battery_size"] *
                        (1 - configuration["calendar_capacity_fade"][12 * simulation_year - 1] / 100))

    # Calculate cycling ageing capacity fade
    cycle_fade = calculate_cycle_ageing(car.dod_ts[timeframe_start:timeframe_end])
    car.battery_size -= cycle_fade[-1]/100


    simulation_results["SOC_ts"] = car.SOC_ts[timeframe_start:timeframe_end]
    simulation_results["distance_ts"] = ts_distance
    simulation_results["el_consumption_ts"] = car.el_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["ice_consumption_ts"] = car.ice_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["charging_consumption_ts"] = car.charging_consumption_ts[timeframe_start:timeframe_end]
    simulation_results["battery_capacity"] = car.capacity_ts[timeframe_start:timeframe_end]

    return simulation_results


def experiment_1():
    settings = {
        "ID": "1.0",
        "simulation_years": 15,
        "ice_consumption_per_100km": 10,
        "battery_size": 12,
        "fuel_tank_size": 50,
        "SOC": 0.2,  # starting SOC
        "el_charging_efficiency": 0.98,
        "el_discharging_efficiency": 0.9,
        "charging_limit": 0.8,
        "charging_interval": [93, 94, 95, 96, 0, 1, 2, 3, 4],
        "c_rate": 0.5,
        "SOC_discharge_limit": 0.25,
        "max_el_power": 100,
        "max_ice_power": 100,
        "tank_level": 30,
        "car_weight": 2000,
        "battery_weight": 800,
        "additional_battery_weight": 100,
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
    configuration["car"] = setup_car(settings)

    # Calculate calendar ageing capacity fade
    calendar_capacity_fade = calculate_calendar_ageing(settings["simulation_years"])
    configuration["calendar_capacity_fade"] = calendar_capacity_fade

    t = configuration["simulation_years"]
    for year in range(1, t + 1):

        simulation_results = run_simulation(configuration, year)

        #print_stats(simulation_results, configuration)
        costs = print_cost_breakdown(configuration,
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


def experiment_2():

    settings = {
        "ID": "1.0",
        "simulation_years": 12,
        "ice_consumption_per_100km": 10,
        "battery_size": 30,
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
        "additional_battery_weight": 100,
        "driving_data":
            r"C:\Users\alexl\Downloads\emobpy_timeseries_original\emobpy_timeseries_original.csv",
        "delta_t": 0.25,
        "end_time": 24 * 365,
        "start_time": 24 * 0,
        "years": 1,
        "plot": True,
        "recharge": True,
        "calc_econ": True,
        "cost_parameters": {
            "vehicle_costs": 30000,  # Costs in Euro
            "build_in_battery_costs": 8000,
            "battery_cell_price": 0.5,  # Costs in Euro/Cell
            "cell_amount": 15000,
            "fuel_costs": 1.8,  # Costs in Euro/Liter
            "electricity_costs": 0.4,  # Costs in Euro/kWh
            "battery_aging_factor": 1.5,
            "maintenance_costs": 0.0096  # Maintenance costs in euro/km
        }
    }
    configuration = setup(settings)
    configuration["car"] = setup_car(settings)

    configuration = modify_driving_profiles_4(configuration)

    simulation_results = run_simulation(configuration)

    print_stats(simulation_results, configuration)

    if settings["plot"]:
        folder_name = create_unique_folder()
        plot_timeseries(configuration["start_time"], configuration["end_time"],
                        configuration["delta_t"], simulation_results, settings["battery_size"], folder_name)

        plot_soc_frequency(simulation_results["SOC_ts"], simulation_results["distance_ts"], settings,
                           settings["battery_size"], folder_name)

def experiment_3():
    enable_pretty_plots()
    battery_capacity = np.array([2,4])
    battery_capacity = np.array([2,3,4,5,6,7,8,9,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

    settings = {
        "ID": "1.0",
        "simulation_years": 12,
        "ice_consumption_per_100km": 7, # liters/100km
        "battery_size": 20,
        "fuel_tank_size": 50,
        "SOC": 0.9,  # starting SOC
        "el_charging_efficiency": 0.98,
        "el_discharging_efficiency": 0.9,
        "charging_limit": 0.9,
        "charging_interval": [93, 94, 95, 96, 0, 1, 2, 3, 4],
        "c_rate": 0.5,
        "SOC_discharge_limit": 0.20,
        "max_el_power": 100,
        "max_ice_power": 100,
        "tank_level": 30,
        "car_weight": 2000,
        "battery_weight": 800,
        "additional_battery_weight": 100,
        "driving_data":
            r"C:\Users\alexl\Downloads\emobpy_timeseries_original\emobpy_timeseries_original.csv",
        "delta_t": 0.25,
        "end_time": 24 * 365,
        "start_time": 24 * 0,
        "years": 1,
        "plot": True,
        "recharge": True,
        "calc_econ": True,
        "cost_parameters": {
            "vehicle_costs": 35000,  # Costs in Euro
            "build_in_battery_capacity": 16,
            "battery_cell_price": 1.5,  # Costs in Euro/Cell
            "cell_capacity": 0.00936,
            "cell_amount": 2136,
            "fuel_costs": 1.8,  # Costs in Euro/Liter
            "electricity_costs": 0.4,  # Costs in Euro/kWh
            "maintenance_costs": 0.0096 * 3  # Maintenance costs in euro/km
        }
    }

    cost_results = {}

    if settings["plot"]:
        folder_name = create_unique_folder()

    for i in range(len(battery_capacity)):
        print("Simulation for", battery_capacity[i], "kWh battery size:")
        experiment_results = {
            "fuel_consumption": [],
            "charging_consumption": [],
        }
        settings["battery_size"] = battery_capacity[i]
        settings["cost_parameters"]["cell_amount"] = battery_capacity[i] / settings["cost_parameters"]["cell_capacity"]

        configuration = setup(settings)
        configuration["car"] = setup_car(settings)

        # Calculate calendar ageing capacity fade
        calendar_capacity_fade = calculate_calendar_ageing(settings["simulation_years"])
        configuration["calendar_capacity_fade"] = calendar_capacity_fade

        # new ts with 50km avg and 400km max
        configuration =  modify_driving_profiles_4(configuration)

        t = configuration["simulation_years"]

        if (settings["plot"]):
            path = os.path.join(folder_name, f"{battery_capacity[i]}_kWh")
            os.makedirs(path, exist_ok=True)

        capacity_fade = np.zeros(t)
        for year in range(1, t + 1):

            simulation_results = run_simulation(configuration, year)
            experiment_results["fuel_consumption"].append(sum(simulation_results["ice_consumption_ts"]))
            experiment_results["charging_consumption"].append(sum(simulation_results["charging_consumption_ts"]))
            capacity_fade[year-1] = simulation_results["battery_capacity"][0]

            if(settings["plot"] and (year == t or year == 1)):
                plot_timeseries(configuration["start_time"], configuration["end_time"],
                                configuration["delta_t"], simulation_results, settings["battery_size"], path, year)
                plot_soc_frequency(simulation_results["SOC_ts"], simulation_results["distance_ts"], settings,
                                   settings["battery_size"], path, year)

        costs = calculate_final_costs(configuration, experiment_results["fuel_consumption"],
                                        experiment_results["charging_consumption"], configuration["cost_parameters"])

        cost_results[battery_capacity[i]] = costs

        if (settings["plot"]):
            plot_capacity_fade(configuration["simulation_years"], capacity_fade, settings["battery_size"], path)


    # Extracting data for plotting
    capacities = list(cost_results.keys())
    yearly_annuity = [cost_results[cap]["yearly_annuity"] for cap in capacities]
    npv_opex = [cost_results[cap]["npv_opex"] for cap in capacities]
    npv_capex = [cost_results[cap]["npv_capex"] for cap in capacities]
    npv = [cost_results[cap]["npv"] for cap in capacities]
    running_costs = [cost_results[cap]["running_costs"] for cap in capacities]
    maintenance_costs = [cost_results[cap]["maintenance_costs"] for cap in capacities]
    electricity_costs = [cost_results[cap]["electricity_costs"] for cap in capacities]
    fuel_costs = [cost_results[cap]["fuel_costs"] for cap in capacities]

    if settings["plot"]:
        plot_econ_results(capacities, yearly_annuity, npv_opex, npv_capex, npv, running_costs, maintenance_costs,
                      electricity_costs, fuel_costs, folder_name, settings["cost_parameters"])

def main():

    #experiment_2()
    experiment_3()

    return 0


if __name__ == "__main__":
    main()
