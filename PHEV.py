import numpy as np

class Car:
    def __init__(self, battery_size=75, fuel_tank_size=50, soc=80, ice_consumption_per_100km=15,
                 soc_discharge_limit=0.2, max_el_power=100, max_ice_power=100, tank_level=30,
                 charging_interval=24, recharge=True, charging_limit=0.9, c_rate=0.5,
                 el_charging_efficiency=0.98, el_discharging_efficiency=0.9, delta_t=0.25, simulation_years=15):

        self.battery_size = battery_size
        self.fuel_tank_size = fuel_tank_size
        self.SOC = soc
        self.tank_level = tank_level
        self.ice_consumption_per_100km = ice_consumption_per_100km
        self.SOC_discharge_limit = soc_discharge_limit
        self.max_el_power = max_el_power
        self.max_ice_power = max_ice_power
        self.el_consumption_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.ice_consumption_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.charging_consumption_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.capacity_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.el_power_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.ice_power_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.SOC_ts = np.zeros(int(simulation_years) * 8760 * int(1/delta_t))
        self.charging_interval = charging_interval
        self.recharge = recharge
        self.charging_limit = charging_limit
        self.c_rate = c_rate
        self.charging_efficiency = el_charging_efficiency
        self.discharge_efficiency = el_discharging_efficiency
        self.charging_cycles = 0

    def calc_consumption(self, consumption, distance, delta_t, time):

        # Calculate average power out of driving profile
        avg_power = consumption/delta_t / self.discharge_efficiency
        el_power = avg_power
        ice_power = 0
        ice_power_share = 0

        # logic to determine which motor handles the power

        # if soc is smaller than soc limit, ice share is 100 %
        if self.SOC <= self.SOC_discharge_limit:

            ice_power = self.max_ice_power
            ice_power_share = 1
            el_power = 0

        # if el motor power is not sufficient, both motors share the power
        elif avg_power > self.max_el_power:

            ice_power = min(avg_power - self.max_el_power, self.max_ice_power)
            ice_power_share = ice_power/self.max_ice_power
            el_power = avg_power

        # power share determines the share of the distance driven by ice
        ice_consumption = ice_power_share * distance / 100 * self.ice_consumption_per_100km

        # calculate consumption out of power, consider efficiency
        el_consumption = el_power * delta_t

        # If  consumption exceeds SOC limit, over-amount will be moved to ice
        if self.SOC - self.SOC_discharge_limit < el_consumption / self.battery_size:

            new_el_consumption = (self.SOC - self.SOC_discharge_limit) * self.battery_size / self.discharge_efficiency
            new_el_consumption = max(new_el_consumption,0)
            el_power = new_el_consumption / delta_t * self.discharge_efficiency
            difference = el_consumption - new_el_consumption

            # if soc is not high enough for consumption of timestep, ice_power_share is increased to support el motor
            ice_power_share = ((ice_power+difference/delta_t)/self.max_ice_power)

            # update ice and el consumption with new shares
            ice_consumption = ice_power_share * distance / 100 * self.ice_consumption_per_100km
            el_consumption = new_el_consumption

        # store power values in ts
        self.el_power_ts[time] = el_power
        self.ice_power_ts[time] = ice_power

        # store consumption in ts
        self.el_consumption_ts[time] = el_consumption
        self.ice_consumption_ts[time] = ice_consumption

        return el_consumption, ice_consumption

    def drive(self, delta_t, time):

        # last consumption entry in consumption ts
        el_consumption = self.el_consumption_ts[time]
        ice_consumption = self.ice_consumption_ts[time]

        # discharging of battery
        self.SOC -= el_consumption / self.battery_size

        # use of fuel
        self.tank_level -= ice_consumption

        # charging of battery
        # checks if car has to be charged
        if self.recharge and time%96 in self.charging_interval and self.charging_limit-self.SOC > 0:

            # checks if battery is going to be fully charged
            if self.charging_limit-self.SOC < self.c_rate * delta_t * self.charging_efficiency:
                self.charging_consumption_ts[time] = (
                        (self.charging_limit-self.SOC)/self.charging_efficiency * self.battery_size)
                self.SOC = self.charging_limit  # battery gets charged to charging_limit

            else:
                self.charging_consumption_ts[time] = (
                        (self.c_rate*delta_t)  * self.battery_size)
                self.SOC += (self.c_rate*delta_t) * self.charging_efficiency
                self.SOC = min(self.SOC, self.charging_limit)

        self.SOC_ts[time] = self.SOC
        self.battery_size = self.battery_size*0.99999
        self.capacity_ts[time] = self.battery_size

        return el_consumption











