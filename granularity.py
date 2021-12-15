from pandas import read_csv
from ds_charts import get_variable_types, choose_grid, HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import HEIGHT
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import subplots, savefig, show

filenames = ['data/NYC_collisions_tabular.csv', 'data/air_quality_tabular.csv']

for filename in filenames:
    data = read_csv(filename)

    short_file_name = filename[5:-12]

    values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
    ##################################################################################

    variables = get_variable_types(data)['Numeric']
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    print(variables)

    if filename != 'data/NYC_collisions_tabular.csv':
        rows, cols = 3, 9
    else:
        rows, cols = 2, 2

    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT+5, rows*HEIGHT+5), squeeze=False)
    i, j = 0, 0
    for n in range(len(variables)):
        axs[i, j].set_title('Histogram for %s'%variables[n])
        axs[i, j].set_xlabel(variables[n])
        axs[i, j].set_ylabel('nr records')
        axs[i, j].hist(data[variables[n]].values, bins=100)
        i, j = (i + 1, 0) if (n+1) % cols == 0 else (i, j + 1)
    savefig(f'images/{short_file_name}/granularity_single.png')
    ##################################################################################

    for variable in variables:
        bins = (3, 5, 10, 100, 1000, 10000)
        fig, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
        for j in range(len(bins)):
            axs[j].set_title('Histogram for %s %d bins'%(variable, bins[j]))
            axs[j].set_xlabel(variable)
            axs[j].set_ylabel('Nr records')
            axs[j].hist(data[variable].values, bins=bins[j])
        savefig(f'images/{short_file_name}/granularity_study_{variable}.png')
        break
    #
    ##################################################################################
    variables = get_variable_types(data)['Numeric']
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    rows = len(variables)
    bins = (10, 100, 1000)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT+5, rows*HEIGHT+5), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            axs[i, j].hist(data[variables[i]].values, bins=bins[j])
    savefig(f'images/{short_file_name}/granularity_study.png')
    # show()
