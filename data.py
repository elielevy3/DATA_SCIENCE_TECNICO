from pandas import DataFrame, read_csv

class Data():
    def __init__(self):
        # read data
        nyc_data = read_csv('data_to_use/nyc_car_crash_without_na.csv')
        air_quality_data = read_csv('data_to_use/air_quality_tabular_without_na.csv')

        # drop columns for nyc data
        values_to_pop_nyc = ['CRASH_DATE', 'VEHICLE_ID', 'PERSON_ID', 'CRASH_TIME']
        for value in values_to_pop_nyc:
            nyc_data.pop(value)
        self.nyc_data = nyc_data

        # drop columns for air quality data
        values_to_pop_air_quality = ['date', 'City_EN', 'GbCity', 'Prov_EN', 'ALARM']
        for value in values_to_pop_air_quality:
            air_quality_data.pop(value)
        self.air_quality_data = air_quality_data
