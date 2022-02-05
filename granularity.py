from pandas import read_csv

filename = 'data/electrical_grid_stability.csv'
data = read_csv(filename)

values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}

