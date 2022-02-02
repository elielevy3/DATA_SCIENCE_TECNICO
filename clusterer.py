import os
import numpy as np
from matplotlib.pyplot import subplots, show, savefig
from ds_charts import choose_grid, plot_clusters, plot_line, \
    compute_centroids, compute_mse, bar_chart, multiple_bar_chart
from sklearn.cluster import KMeans
from data import Data
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from featureExtractor import FeatureExtractor
from featureSelector import FeatureSelector
from sklearn import preprocessing
import pandas as pd


class Clusterer():
    def __init__(self, data_set='air_quality'):
        data = Data()
        self.data_set = data_set
        self.data = data.get_air_quality_data() if data_set == 'air_quality' else data.get_nyc_data()
        self.data = self.data.sample(n=5000)
        self.data_original = self.data

        if not os.path.exists(f'images_nyc/lab8'):
            os.makedirs(f'images_nyc/lab8')
        if not os.path.exists(f'images_air_quality/lab8'):
            os.makedirs(f'images_air_quality/lab8')

        self.output_path = f'images_{self.data_set}/lab8'

    def apply_pca(self, n_components=4):
        featureExtractor = FeatureExtractor(data_set=self.data_set, data=self.data)

        self.data = featureExtractor.apply_pca(n_components=n_components)

        if not os.path.exists(f'images_{self.data_set}/lab8/pca'):
            os.makedirs(f'images_{self.data_set}/lab8/pca')
        self.output_path += '/pca'

    def apply_feature_selection(self):
        featureSelector = FeatureSelector(self.data_set, self.data)
        self.data = featureSelector.drop_redundant_and_low_variance_variables()

    def normalize_data(self):
        columns = list(self.data.columns)
        self.data = preprocessing.normalize(self.data, norm='l2')
        self.data = pd.DataFrame(self.data, columns=columns)

    def cluster_with_kmeans(self, show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None:
            N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = choose_grid(len(N_CLUSTERS))

        mse: list = []
        sc: list = []
        fig, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = KMeans(n_clusters=k)
            estimator.fit(self.data)
            mse.append(estimator.inertia_)
            sc.append(silhouette_score(self.data, estimator.labels_))
            plot_clusters(self.data, v2, v1, estimator.labels_.astype(float), estimator.cluster_centers_, k,
                          f'KMeans k={k}',
                          ax=axs[i, j])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        savefig(f'{self.output_path}/kMeans.png')

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'images_{self.data_set}/lab8/kMeans_MSE_SC.png')
        if show_chart:
            show()

        print(f' Clustering with kmeans of {self.data_set} finished')

        return N_CLUSTERS, mse, sc

    def cluster_with_kmeans_mse_sc(self, show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None:
            N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        N_CLUSTERS, mse, sc = self.cluster_with_kmeans(show_chart=False, N_CLUSTERS=N_CLUSTERS, v1=v1, v2=v2)

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        plot_line(N_CLUSTERS, mse, title='KMeans MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        plot_line(N_CLUSTERS, sc, title='KMeans SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'{self.output_path}/kMeans_MSE_SC.png')
        if show_chart:
            show()

        print(f' Clustering with kmeans and save mse and sc of {self.data_set} finished')

    def cluster_density_based_eps(self, show_chart=False, v1=0, v2=4):
        N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = choose_grid(len(N_CLUSTERS))

        EPS = [2.5, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        mse: list = []
        sc: list = []
        rows, cols = choose_grid(len(EPS))
        _, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        i, j = 0, 0
        for n in range(len(EPS)):
            estimator = DBSCAN(eps=EPS[n], min_samples=2)
            estimator.fit(self.data)
            labels = estimator.labels_
            k = len(set(labels)) - (1 if -1 in labels else 0)
            if k > 1:
                centers = compute_centroids(self.data, labels)
                mse.append(compute_mse(self.data.values, labels, centers))
                sc.append(silhouette_score(self.data, labels))
                plot_clusters(self.data, v2, v1, labels.astype(float), estimator.components_, k,
                              f'DBSCAN eps={EPS[n]} k={k}', ax=axs[i, j])
                i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
            else:
                mse.append(0)
                sc.append(0)

        savefig(f'{self.output_path}/ds_based_clustering_max_dist_impact.png')
        if show_chart:
            show()

        print(f' Clustering with ds based clustering of {self.data_set} finished')
        return EPS, mse, sc

    def cluster_density_based_eps_mse_sc(self, show_chart=False, v1=0, v2=1):

        EPS, mse, sc = self.cluster_density_based_eps(show_chart=False, v1=v1, v2=v2)

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        plot_line(EPS, mse, title='DBSCAN MSE', xlabel='eps', ylabel='MSE', ax=ax[0, 0])
        plot_line(EPS, sc, title='DBSCAN SC', xlabel='eps', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'{self.output_path}/ds_based_clustering_max_dist_ms_sc.png')
        if show_chart:
            show()
        print(f' Clustering with ds based clustering and save mse and sc of {self.data_set} finished')

    def cluster_ds_based_with_different_metrics(self, show_chart=False, v1=0, v2=4):
        METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']
        distances = []
        for m in METRICS:
            dist = np.mean(np.mean(squareform(pdist(self.data.values, metric=m))))
            distances.append(dist)

        print('AVG distances among records', distances)
        distances[0] *= 0.6
        distances[1] = 80
        distances[2] *= 0.6
        distances[3] *= 0.1
        distances[4] *= 0.15
        print('CHOSEN EPS', distances)

        mse: list = []
        sc: list = []
        rows, cols = choose_grid(len(METRICS))
        _, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        i, j = 0, 0
        for n in range(len(METRICS)):
            estimator = DBSCAN(eps=distances[n], min_samples=2, metric=METRICS[n])
            estimator.fit(self.data)
            labels = estimator.labels_
            k = len(set(labels)) - (1 if -1 in labels else 0)
            if k > 1:
                centers = compute_centroids(self.data, labels)
                mse.append(compute_mse(self.data.values, labels, centers))
                sc.append(silhouette_score(self.data, labels))
                plot_clusters(self.data, v2, v1, labels.astype(float), estimator.components_, k,
                              f'DBSCAN metric={METRICS[n]} eps={distances[n]:.2f} k={k}', ax=axs[i, j])
            else:
                mse.append(0)
                sc.append(0)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        savefig(f'{self.output_path}/ds_based_clustering_different_metrics.png')
        if show_chart:
            show()

        print(f' Clustering with different metrics for ds based clustering of {self.data_set} finished')

        return METRICS, mse, sc

    def cluster_ds_based_with_different_metrics_mse_sc(self, show_chart=False, v1=0, v2=1):
        METRICS, mse, sc = self.cluster_ds_based_with_different_metrics(show_chart=False, v1=v1, v2=v2)

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        bar_chart(METRICS, mse, title='DBSCAN MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        bar_chart(METRICS, sc, title='DBSCAN SC', xlabel='metric', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'{self.output_path}/ds_based_clustering_different_metrics_mse_sc.png')
        if show_chart:
            show()

        print(f' Clustering with different metrics for ds based clustering and save mse '
              f'and sc of {self.data_set} finished')

    def cluster_with_em(self, show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None:
            N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

        rows, cols = choose_grid(len(N_CLUSTERS))

        mse: list = []
        sc: list = []
        _, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = GaussianMixture(n_components=k)
            estimator.fit(self.data)
            labels = estimator.predict(self.data)
            mse.append(compute_mse(self.data.values, labels, estimator.means_))
            sc.append(silhouette_score(self.data, labels))
            plot_clusters(self.data, v2, v1, labels.astype(float), estimator.means_, k,
                          f'EM k={k}', ax=axs[i, j])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        savefig(f'{self.output_path}/clustering_with_em.png')
        if show_chart:
            show()

        print(f' Clustering with different em of {self.data_set} finished')

        return N_CLUSTERS, mse, sc

    def cluster_with_em_mse_sc(self, show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None:
            N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        N_CLUSTERS, mse, sc = self.cluster_with_em(show_chart=False, N_CLUSTERS=N_CLUSTERS, v1=v1, v2=v2)

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        plot_line(N_CLUSTERS, mse, title='EM MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        plot_line(N_CLUSTERS, sc, title='EM SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'{self.output_path}/clustering_with_em_mse_sc.png')
        if show_chart:
            show()

        print(f' Clustering with different em and save mse and sc of {self.data_set} finished')

    def cluster_hierarchical(self, show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None:
            N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        rows, cols = choose_grid(len(N_CLUSTERS))

        mse: list = []
        sc: list = []
        rows, cols = choose_grid(len(N_CLUSTERS))
        _, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        i, j = 0, 0
        for n in range(len(N_CLUSTERS)):
            k = N_CLUSTERS[n]
            estimator = AgglomerativeClustering(n_clusters=k)
            estimator.fit(self.data)
            labels = estimator.labels_
            centers = compute_centroids(self.data, labels)
            mse.append(compute_mse(self.data.values, labels, centers))
            sc.append(silhouette_score(self.data, labels))
            plot_clusters(self.data, v2, v1, labels, centers, k, f'Hierarchical k={k}', ax=axs[i, j])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

        savefig(f'{self.output_path}/cluster_hierarchical.png')
        if show_chart:
            show()

        print(f' Clustering hierarchical of {self.data_set} finished')

        return N_CLUSTERS, mse, sc

    def cluster_hierarchical_mse_sc(self, data_set='air_quality', show_chart=False, N_CLUSTERS=None, v1=0, v2=1):
        if N_CLUSTERS is None: N_CLUSTERS = [2, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

        N_CLUSTERS, mse, sc = self.cluster_hierarchical(show_chart=False, N_CLUSTERS=N_CLUSTERS, v1=v1, v2=v2)

        fig, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        plot_line(N_CLUSTERS, mse, title='Hierarchical MSE', xlabel='k', ylabel='MSE', ax=ax[0, 0])
        plot_line(N_CLUSTERS, sc, title='Hierarchical SC', xlabel='k', ylabel='SC', ax=ax[0, 1], percentage=True)

        savefig(f'{self.output_path}/cluster_hierarchical_mse_sc.png')
        if show_chart:
            show()

        print(f' Clustering hierarchical and save mse and sc of {self.data_set} finished')

    def cluster_hierarchical_with_different_metrics(self, show_chart=False, METRICS=None, LINKS=None, v1=0, v2=1):
        if LINKS is None:
            LINKS = ['complete', 'average']
        if METRICS is None:
            METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']

        k = 3

        values_mse = {}
        values_sc = {}
        rows = len(METRICS)
        cols = len(LINKS)
        _, axs = subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        for i in range(len(METRICS)):
            mse: list = []
            sc: list = []
            m = METRICS[i]
            for j in range(len(LINKS)):
                link = LINKS[j]
                estimator = AgglomerativeClustering(n_clusters=k, linkage=link, affinity=m)
                estimator.fit(self.data)
                labels = estimator.labels_
                centers = compute_centroids(self.data, labels)
                mse.append(compute_mse(self.data.values, labels, centers))
                sc.append(silhouette_score(self.data, labels))
                plot_clusters(self.data, v2, v1, labels, centers, k, f'Hierarchical k={k} metric={m} link={link}',
                              ax=axs[i, j])
            values_mse[m] = mse
            values_sc[m] = sc

        savefig(f'{self.output_path}/cluster_hierarchical_with_different_metrics.png')
        if show_chart:
            show()

        print(f' Clustering hierarchical with different metrics of {self.data_set} finished')

        return values_mse, values_sc

    def cluster_hierarchical_with_different_metrics_mse_sc(self, show_chart=False, METRICS=None, LINKS=None, v1=0,
                                                           v2=1):
        if LINKS is None:
            LINKS = ['complete', 'average']
        if METRICS is None:
            METRICS = ['euclidean', 'cityblock', 'chebyshev', 'cosine', 'jaccard']

        values_mse, values_sc = self.cluster_hierarchical_with_different_metrics(METRICS=METRICS, LINKS=LINKS,
                                                                                 show_chart=False,
                                                                                 v1=v1, v2=v2)

        _, ax = subplots(1, 2, figsize=(6, 3), squeeze=False)
        multiple_bar_chart(LINKS, values_mse, title=f'Hierarchical MSE', xlabel='metric', ylabel='MSE', ax=ax[0, 0])
        multiple_bar_chart(LINKS, values_sc, title=f'Hierarchical SC', xlabel='metric', ylabel='SC', ax=ax[0, 1],
                           percentage=True)

        savefig(f'{self.output_path}/cluster_hierarchicalv_with_different_metrics_mse_sc.png')
        if show_chart:
            show()

        print(f' Clustering hierarchical with different metrics and save mse and sc of {self.data_set} finished')

    def run_all_cluster_methods(self, v1=0, v2=1):
        self.cluster_with_kmeans_mse_sc(v1=v1, v2=v2)
        self.cluster_with_em_mse_sc(v1=v1, v2=v2)
        self.cluster_ds_based_with_different_metrics_mse_sc(v1=v1, v2=v2)
        self.cluster_hierarchical_with_different_metrics_mse_sc(v1=v1, v2=v2)


if __name__ == "__main__":
    # airQualityClusterer without PCA and with PCA
    # without PCA
    # aqClusterWithoutPca = Clusterer(data_set='air_quality')
    # aqClusterWithoutPca.apply_feature_selection()
    # v1, v2 = aqClusterWithoutPca.data.columns.get_loc("NO2_Mean"), \
    #          aqClusterWithoutPca.data.columns.get_loc("O3_Mean")
    #
    # aqClusterWithoutPca.run_all_cluster_methods(v1=v1, v2=v2)
    #
    # # with PCA
    # aqClusterWithPca = Clusterer(data_set='air_quality')
    # aqClusterWithPca.apply_feature_selection()
    # aqClusterWithPca.apply_pca(n_components=4)
    # aqClusterWithPca.run_all_cluster_methods()

    # nycClusterer without PCA and with PCA
    # without PCA
    nycClustererWithoutPca = Clusterer(data_set='nyc')
    nycClustererWithoutPca.apply_feature_selection()
    nycClustererWithoutPca.normalize_data()
    v1, v2 = nycClustererWithoutPca.data.columns.get_loc("PERSON_AGE"), \
             nycClustererWithoutPca.data.columns.get_loc("EMOTIONAL_STATUS")
    nycClustererWithoutPca.run_all_cluster_methods(v1=v1, v2=v2)

    # with PCA
    nycClustererWithPca = Clusterer(data_set='nyc')
    nycClustererWithPca.apply_feature_selection()
    nycClustererWithPca.apply_pca(n_components=3)
    nycClustererWithPca.run_all_cluster_methods()