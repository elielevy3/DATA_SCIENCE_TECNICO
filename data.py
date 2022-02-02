from pandas import read_csv

class Data():
    def __init__(self):
        # read data
        nyc_data = read_csv('data_to_use/nyc_car_crash_without_na.csv')
        air_quality_data = read_csv('data_to_use/air_quality_tabular_without_na.csv')

        # drop columns for nyc data
        values_to_pop_nyc = ['CRASH_DATE', 'VEHICLE_ID', 'PERSON_ID', 'CRASH_TIME', 'UNIQUE_ID', 'COLLISION_ID']
        for value in values_to_pop_nyc:
            nyc_data.pop(value)
        self.nyc_data = nyc_data
        self.nyc_data = self.nyc_data.loc[:, ~self.nyc_data.columns.str.contains('^Unnamed')]

        # drop columns for air quality data
        values_to_pop_air_quality = ['date', 'City_EN', 'GbCity', 'Prov_EN', 'ALARM']
        for value in values_to_pop_air_quality:
            air_quality_data.pop(value)
        self.air_quality_data = air_quality_data
        self.air_quality_data = self.air_quality_data.loc[:, ~self.air_quality_data.columns.str.contains('^Unnamed')]

    def get_nyc_data(self):
        return self.nyc_data

    def get_air_quality_data(self):
        return self.air_quality_data


