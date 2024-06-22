import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from datetime import datetime
import os
import re
import seaborn as sns
import matplotlib.ticker
from matplotlib import colors, lines

BLUE = '#2CBDFE'
GREEN = '#47DBCD'
PINK = '#F3A0F2'
PURPLE = '#9D2EC5'
VIOLET = '#661D98'
AMBER = '#F5B14C'


def enable_pretty_plots():
    sns.set(font='Franklin Gothic Book', rc={
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'grid.color': (0.85, 0.85, 0.85),
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'axes.labelcolor': 'dimgrey',
        # 'axes.spines.right': False,
        # 'axes.spines.top': False,
        'figure.facecolor': 'white',
        'legend.framealpha': 1,
        'legend.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        # 'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        # 'ytick.left': False,
        'ytick.right': False,

    })

    sns.set_context("notebook", rc={
        "font.size": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        'figure.titlesize': "large"
    })

    plt.rcParams['savefig.pad_inches'] = 0.2

    color_list = [BLUE, PINK, GREEN, AMBER, PURPLE, VIOLET]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)

    gradient_map = ['#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8',
                    '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
                    '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
                    '#568ae6', '#5986e4', '#5c81e2', '#607de0',
                    '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
                    '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
                    '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
                    '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
                    '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
                    '#a924b7', '#ac20b5', '#b01bb3', '#b317b1']

    plt.colormaps.register(colors.ListedColormap(gradient_map, 'gradient_map'))



def plot_timeseries(start_time, end_time, time_step, data_dict, battery_size, base_dir, year=1):
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

    # Create new folder with the next ID
    folder_name = "TimeSeries"
    full_folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    # Save the figure
    file_name = f"TS_{year}_{battery_size}kWh.png"
    file_path = os.path.join(full_folder_path, file_name)
    plt.savefig(file_path)
    plt.close()


def plot_soc_frequency(soc_ts, distance_ts, settings, battery_size, base_dir, year=1):

    # Plot setup
    plt.figure(figsize=(10, 12))

    # First plot: Histogram of state of charge (soc_ts)
    mask = distance_ts != 0
    soc_lower_limit = settings["SOC_discharge_limit"]
    soc_upper_limit = settings["charging_limit"]
    plt.hist(soc_ts[mask], bins=50, weights=np.ones(len(soc_ts[mask])) / len(soc_ts[mask]))
    plt.axvline(soc_lower_limit, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(soc_upper_limit, color='g', linestyle='dashed', linewidth=1)
    plt.title('Histogram of State of Charge')
    plt.xlabel('State of Charge')
    plt.ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Create folder and save figure
    folder_name = "Frequency"
    full_folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    file_name = f"F_{year}_{battery_size}kWh.png"
    file_path = os.path.join(full_folder_path, file_name)
    plt.savefig(file_path)

    # Close plot
    plt.close()



def plot_capacity_fade(year, capacity_fade, battery_size, base_dir):

    years = range(0, year)

    # Plotting
    plt.figure(figsize=(10, 6))

    plt.plot(years, capacity_fade, marker='o', linestyle='-',
             markersize=4, linewidth=1)  # Adjust markersize and linewidth as needed

    plt.xlabel('Time')
    plt.ylabel('Capacity Fade')
    plt.title(f'Capacity Fade Over Time for Battery Size {battery_size} kWh')
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d-%m %H:%M"))
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Create new folder if not exists
    folder_name = "CapacityFade"
    full_folder_path = os.path.join(base_dir, folder_name)
    os.makedirs(full_folder_path, exist_ok=True)

    # Save the figure
    file_name = f"CapacityFade_{battery_size}kWh.png"
    file_path = os.path.join(full_folder_path, file_name)
    plt.savefig(file_path)
    plt.close()


def plot_econ_results(capacities, yearly_annuity, npv_opex, npv_capex, npv, running_costs, maintenance_costs,
                      electricity_costs, fuel_costs, path, cost_parameters):

    now = datetime.now().strftime("%Y%m%d_%H%M")
    os.makedirs(now, exist_ok=True)

    # Creating the 2x2 multiplot
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))

    # Plotting Yearly Annuity
    axs[0, 0].plot(capacities, yearly_annuity, marker='o')
    axs[0, 0].set_title('Yearly Annuity')
    axs[0, 0].set_xlabel('Capacity')
    axs[0, 0].set_ylabel('Yearly Annuity')
    axs[0, 0].grid(True, axis='y')
    axs[0, 0].set_xticks(capacities)  # Set x-axis ticks to the unique capacities

    x_max = max(capacities)
    y_max = max(yearly_annuity)
    text_x = min(capacities) + 0.025 * (x_max - min(capacities))  # Adjust the position as needed
    text_y = y_max - 0.025 * (y_max - min(yearly_annuity))  # Adjust the position as needed

    axs[0, 0].text(text_x, text_y,
                   f"Fuel Costs: {cost_parameters['fuel_costs']}€/l\n"
                   f"Electricity Costs: {cost_parameters['electricity_costs']}€/kWh\n"
                   f"Battery Cell Costs: {cost_parameters['battery_cell_price']}€\n"
                   f"Maintenance Costs: {cost_parameters['maintenance_costs']}€/km\n"
                   f"PHEV Costs: {cost_parameters['vehicle_costs']}€\n",
                   fontsize=10, ha='left', va='top')

    # Plotting NPV as stacked bar plot
    axs[0, 1].bar(capacities, npv_opex, label='NPV Opex', bottom=npv_capex)
    axs[0, 1].bar(capacities, npv_capex, label='NPV Capex')
    axs[0, 1].set_title('NPV (Opex + Capex)')
    axs[0, 1].set_xlabel('Capacity')
    axs[0, 1].set_ylabel('NPV')
    axs[0, 1].grid(True, axis='y')
    axs[0, 1].legend()
    axs[0, 1].set_xticks(capacities)  # Set x-axis ticks to the unique capacities

    # Plotting Running Costs as stacked bar plot
    axs[1, 0].bar(capacities, maintenance_costs, label='Maintenance Costs', color='g',
                  bottom=[i + j for i, j in zip(electricity_costs, fuel_costs)])
    axs[1, 0].bar(capacities, electricity_costs, label='Electricity Costs', color='b',
                  bottom=fuel_costs)
    axs[1, 0].bar(capacities, fuel_costs, label='Fuel Costs', color='r')
    axs[1, 0].set_title('Running Costs Breakdown')
    axs[1, 0].set_xlabel('Capacity')
    axs[1, 0].set_ylabel('Running Costs')
    axs[1, 0].grid(True, axis='y')
    axs[1, 0].legend()
    axs[1, 0].set_xticks(capacities)  # Set x-axis ticks to the unique capacities

    # Adjust layout to prevent overlap
    fig.tight_layout()

    plt.savefig(os.path.join(path, 'EconResults.png'))
