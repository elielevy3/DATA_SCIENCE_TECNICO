from matplotlib.pyplot import figure, title, savefig, show
from seaborn import heatmap
from ds_charts import bar_chart
import os
from data import Data


class FeatureSelector():
    def __init__(self, data_set, data):

        self.data = data
        self.data_set = data_set

        self.threshold = 0.9
        self.variance_threshold = 0.1

        if not os.path.exists(f'images_nyc/featureEngineering'):
            os.makedirs(f'images_nyc/featureEngineering')
        if not os.path.exists(f'images_air_quality/featureEngineering'):
            os.makedirs(f'images_air_quality/featureEngineering')

        self.output_path = f'images_{self.data_set}/featureEngineering'

    def select_redundant(self):
        corr_mtx = self.data.corr()
        if corr_mtx.empty:
            return {}

        corr_mtx = abs(corr_mtx)
        vars_2drop = {}
        for el in corr_mtx.columns:
            el_corr = (corr_mtx[el]).loc[corr_mtx[el] >= self.threshold]
            if len(el_corr) == 1:
                corr_mtx.drop(labels=el, axis=1, inplace=True)
                corr_mtx.drop(labels=el, axis=0, inplace=True)
            else:
                vars_2drop[el] = el_corr.index

            if len(vars_2drop) > 1:
                print(f'redundant variables dropped {vars_2drop}')
        return vars_2drop, corr_mtx

    def show_correlation_matrix(self, show_figure=False):
        vars_2drop, corr_mtx = self.select_redundant(self.data.corr(), self.threshold)
        if corr_mtx.empty:
            raise ValueError('Matrix is empty.')

        figure(figsize=[10, 10])
        heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=False, cmap='Blues')
        title('Filtered Correlation Analysis')

        savefig(f'images/filtered_correlation_analysis_{self.threshold}.png')
        if show_figure:
            show()

    def drop_redundant(self):
        vars_2drop, corr_mtx = self.select_redundant()
        sel_2drop = []
        for key in vars_2drop.keys():
            if key not in sel_2drop:
                for r in vars_2drop[key]:
                    if r != key and r not in sel_2drop:
                        sel_2drop.append(r)
        if len(sel_2drop) > 1:
            print('Variables to drop', sel_2drop)
        df = self.data.copy()
        for var in sel_2drop:
            df.drop(labels=var, axis=1, inplace=True)
        return df

    def drop_redundant_and_low_variance_variables(self):
        data = self.drop_redundant()

        lst_variables = []
        lst_variances = []
        for el in data.columns:
            value = data[el].var()
            # print(value)
            if value <= self.variance_threshold:
                lst_variables.append(el)
                lst_variances.append(value)

        if len(lst_variables) > 1:
            print('Variables to drop variance', lst_variables)

        figure(figsize=[10, 4])

        if len(lst_variables) > 1:
            bar_chart(lst_variables, lst_variances, title='Variance analysis', xlabel='variables', ylabel='variance')
            savefig(f'{self.output_path}/filtered_variance_analysis.png')

        for var in lst_variables:
            data.drop(labels=var, axis=1, inplace=True)
        return data

if __name__ == '__main__':
    # air quality
    data = Data()

    featureSelector = FeatureSelector(data_set='air_quality', data=data)


    var2_drop, _ = featureSelector.select_redundant()
    data_before_feature_selection = featureSelector.data
    data_after_feature_selection = featureSelector.drop_redundant_and_low_variance_variables()



