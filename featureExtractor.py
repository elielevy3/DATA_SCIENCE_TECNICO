from matplotlib.pyplot import figure, xlabel, ylabel, scatter, show, subplots
from data import Data
from sklearn.decomposition import PCA
from featureSelector import FeatureSelector
from numpy.linalg import eig
from matplotlib.pyplot import gca, title
import pandas as pd
from sklearn import preprocessing
import os


class FeatureExtractor():
    def __init__(self, data_set, data=None, n_components=5):
        if data is not None:
            self.data = data
            self.data_set = data_set
        else:
            data = Data()
            self.data_set = data_set
            self.data = data.air_quality_data if data_set == 'air_quality' else data.nyc_data
            self.data = self.data

        self.data = self.data
        self.n_components = n_components
        self.featureSelector = FeatureSelector(data_set=self.data_set, data=self.data)

        if not os.path.exists(f'images_nyc/featureEngineering'):
            os.makedirs(f'images_nyc/featureEngineering')
        if not os.path.exists(f'images_air_quality/featureEngineering'):
            os.makedirs(f'images_air_quality/featureEngineering')

        self.output_path = f'images_{self.data_set}/featureEngineering'

    def get_rid_of_redundant_features(self):
        data = self.featureSelector.drop_redundant_and_low_variance_variables()

        print(data)

    def show(self, show_figure=False):
        variables = self.data.columns.values
        eixo_y = 0
        eixo_z = 1

        figure()
        xlabel(variables[eixo_y])
        ylabel(variables[eixo_z])
        scatter(self.data.iloc[:, eixo_y], self.data.iloc[:, eixo_z])

        if show_figure:
            show()

        return variables, eixo_y, eixo_z

    def show_variance_ratio(self, show_figure=False):
        mean = (self.data.mean(axis=0)).tolist()
        centered_data = self.data - mean
        cov_mtx = centered_data.cov()
        eigvals, eigvecs = eig(cov_mtx)

        pca = PCA()
        pca.fit(centered_data)
        PC = pca.components_
        var = pca.explained_variance_

        # data = pd.DataFrame(pca.components_, columns=self.data.columns, index=['PC-1', 'PC-2', 'PC-3', 'PC-4'])

        # PLOT EXPLAINED VARIANCE RATIO
        fig = figure(figsize=(4, 4))
        title('Explained variance ratio')
        xlabel('PC')
        ylabel('ratio')
        x_values = [str(i) for i in range(1, len(pca.components_) + 1)]
        bwidth = 0.5
        ax = gca()
        ax.set_xticklabels(x_values)
        ax.set_ylim(0.0, 1.0)
        ax.bar(x_values, pca.explained_variance_ratio_, width=bwidth)
        ax.plot(pca.explained_variance_ratio_)

        for i, v in enumerate(pca.explained_variance_ratio_):
            ax.text(i, v + 0.05, f'{v * 100:.1f}', ha='center', fontweight='bold')

        if show_figure:
            show()

        return pca

    def apply_pca(self, show_figure=False, n_components=5):
        variables, eixo_y, eixo_z  = self.show(False)

        mean = (self.data.mean(axis=0)).tolist()
        # centered_data = self.data - mean

        centered_data = preprocessing.normalize(self.data, norm='l2')

        pca = PCA(n_components=n_components)
        pca.fit(centered_data)

        # rec = pd.DataFrame(pca.components_, columns=self.scaled.columns, index=['PC-1', 'PC-2'])

        transf = pca.transform(self.data)
        cols = []

        for i in range(1, n_components + 1):
            cols.append(f'pc{i}')

        df_transformed = pd.DataFrame(data=transf, columns=cols)

        _, axs = subplots(1, 2, figsize=(2 * 5, 1 * 5), squeeze=False)
        axs[0, 0].set_xlabel(variables[eixo_y])
        axs[0, 0].set_ylabel(variables[eixo_z])
        axs[0, 0].scatter(self.data.iloc[:, eixo_y], self.data.iloc[:, eixo_z])

        axs[0, 1].set_xlabel('PC1')
        axs[0, 1].set_ylabel('PC2')
        axs[0, 1].scatter(transf[:, 0], transf[:, 1])

        if show_figure:
            show()

        return df_transformed

if __name__ == "__main__":
    featureExtractor = FeatureExtractor('nyc')


    featureExtractor.show_variance_ratio(show_figure=True)
    # featureExtractor.get_rid_of_redundant_features()
    # transformed_data = featureExtractor.apply_pca()

    # featureExtractor.show_variance_ratio(show_figure=True)

    # featureExtractor = FeatureExtractor('air_quality')
    # featureExtractor.show_variance_ratio(show_figure=True)
    # featureExtractor.get_rid_of_redundant_features()
    # featureExtractor.show_variance_ratio(show_figure=True)
    # transformed_data = featureExtractor.apply_pca()