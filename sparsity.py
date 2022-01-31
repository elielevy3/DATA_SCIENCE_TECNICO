from pandas import read_csv
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import figure, savefig, title
from seaborn import heatmap

register_matplotlib_converters()

filenames = ['data_original/NYC_collisions_tabular.csv', 'data_original/air_quality_tabular.csv']


for filename in filenames:
    short_file_name = filename[5:-12]
    data = read_csv(filename)

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    fig, axs = subplots(rows, cols, figsize=(cols*HEIGHT+5, rows*HEIGHT+5), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    savefig(f'images/{short_file_name}/sparsity_study_numeric.png')
    # show()

    register_matplotlib_converters()
    data = read_csv(filename)

    numeric_vars = get_variable_types(data)['Numeric']
    if [] == numeric_vars:
        raise ValueError('There are no numeric variables.')

    rows, cols = len(numeric_vars) - 1, len(numeric_vars) - 1
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT+5, rows * HEIGHT+5), squeeze=False)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i + 1, len(numeric_vars)):
            var2 = numeric_vars[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(data[var1], data[var2])
    savefig(f'images/{short_file_name}/sparsity_study_numeric.png')
    #show()

    symbolic_vars = get_variable_types(data)['Symbolic']
    if [] == symbolic_vars:
        raise ValueError('There are no symbolic variables.')

    rows, cols = len(symbolic_vars) - 1, len(symbolic_vars) - 1
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT+5, rows * HEIGHT+5), squeeze=False)
    for i in range(len(symbolic_vars)):
        var1 = symbolic_vars[i]
        for j in range(i + 1, len(symbolic_vars)):
            var2 = symbolic_vars[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(data[var1], data[var2])
    savefig(f'images/{short_file_name}/sparsity_study_symbolic.png')

    corr_mtx = abs(data.corr())

    fig = figure(figsize=[12, 12])

    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    title('Correlation analysis')
    savefig(f'images/{short_file_name}/correlation_analysis.png')