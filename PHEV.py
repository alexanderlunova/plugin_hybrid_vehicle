class Car:
    def __init__(self, battery_size=75, fuel_tank_size=50, SOC=80, ice_consumption_per_100km=15,
                 SOC_limit = 0.2, max_el_power = 100, max_ice_power = 100, tank_level = 30,
                 charging_interval = 24, recharge = True, charging_limit = 0.9):

        self.battery_size = battery_size
        self.fuel_tank_size = fuel_tank_size
        self.SOC = SOC
        self.tank_level = tank_level
        self.ice_consumption_per_100km = ice_consumption_per_100km
        self.SOC_limit = SOC_limit
        self.max_el_power = max_el_power
        self.max_ice_power = max_ice_power
        self.el_consumption_ts = []
        self.ice_consumption_ts = []
        self.charging_interval = charging_interval
        self.recharge = recharge
        self.charging_limit = charging_limit




    def calc_consumption(self, consumption, distance, delta_t):

        avg_power = consumption/delta_t
        el_power = avg_power
        ice_power = 0
        ice_power_share = 0

        if (self.SOC < self.SOC_limit):
            ice_power = self.max_ice_power
            ice_power_share = 1
            el_power = 0
        elif (avg_power > self.max_el_power):
            ice_power = min(avg_power - self.max_el_power, self.max_ice_power)
            ice_power_share = ice_power/self.max_ice_power
            el_power = avg_power

        ice_consumption = ice_power_share * distance / 100 * self.ice_consumption_per_100km
        el_consumption = el_power * delta_t


        #If  consumption exceeds SOC limit, overamount will be moved to ice
        if (self.SOC - self.SOC_limit < el_consumption / self.battery_size):
            new_el_consumption = (self.SOC - self.SOC_limit) * self.battery_size
            new_el_power = new_el_consumption / delta_t
            difference = el_consumption - new_el_consumption
            ice_power_share = ((ice_power+difference/delta_t)/self.max_ice_power)

            ice_consumption = ice_power_share * distance / 100 * self.ice_consumption_per_100km
            el_consumption = new_el_power * delta_t

        return el_consumption, ice_consumption

    def drive(self, el_consumption, ice_consumption, distance, delta_t, t):

        self.SOC -= el_consumption / self.battery_size
        self.tank_level -= ice_consumption
        self.ice_consumption_ts.append(ice_consumption)
        self.el_consumption_ts.append(el_consumption)

        if(self.recharge and t%self.charging_interval == 0):
            self.SOC = self.charging_limit

        return el_consumption






