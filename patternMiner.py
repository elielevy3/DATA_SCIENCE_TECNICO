from matplotlib.pyplot import figure, show, subplots, savefig
from mlxtend.frequent_patterns import apriori, association_rules
from pandas import DataFrame
import os

from sklearn.preprocessing import KBinsDiscretizer

from data import Data
from ds_charts import plot_line, multiple_line_chart, dummify


class PatternMiner():
    def __init__(self, data_set='air_quality'):
        data = Data()
        self.data_set = data_set
        self.data = data.air_quality_data if data_set == 'air_quality' else data.nyc_data
        self.data = self.data.sample(40)

        # self.data_disc = DataFrame(
        #     KBinsDiscretizer(n_bins=4, encode='ordinal',
        #                      strategy='uniform').fit_transform(self.data), columns=self.data.columns)
        #
        # self.data_disc_quantile = DataFrame(
        #     KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile').fit_transform(self.data),
        #     columns=self.data.columns)

        self.data = dummify(self.data, self.data.columns)

        self.output_path = f'images_nyc/lab9'

        # self.data = eq_width
        self.MIN_SUP = 0.5
        self.var_min_sup = [0.2, 0.1] + [i * self.MIN_SUP for i in range(100, 0, -10)]


    def get_columns(self):
        mean_cols = []
        std_cols = []
        min_cols = []
        max_cols = []
        for col in self.data_disc.columns:
            if 'Mean' in col:
                mean_cols.append(col)
            elif 'Std' in col:
                std_cols.append(col)
            elif 'Min' in col:
                min_cols.append(col)
            elif 'Max' in col:
                max_cols.append(col)

        cols = mean_cols + std_cols
        return cols

    def show_patterns(self, show_figure=False):
        patterns: DataFrame = apriori(self.data, min_support=self.MIN_SUP, use_colnames=True, verbose=True)
        print(len(patterns), 'patterns')
        nr_patterns = []
        for sup in self.var_min_sup:
            pat = patterns[patterns['support'] >= sup]
            nr_patterns.append(len(pat))

        figure(figsize=(6, 4))
        plot_line(self.var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')

        savefig(f'{self.output_path}/show_pcas.png')

        if show_figure:
            show()

        return patterns

    def computeNumberOfRules(self):
        patterns = self.show_patterns()

        MIN_CONF: float = 0.1
        rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
        print(f'\tfound {len(rules)} rules')

    def plot_top_rules(self, rules: DataFrame, metric: str, per_metric: str, show_figure=False) -> None:
        _, ax = subplots(figsize=(6, 3))
        ax.grid(False)
        ax.set_axis_off()
        ax.set_title(f'TOP 10 per Min {per_metric} - {metric}', fontweight="bold")
        text = ''
        cols = ['antecedents', 'consequents']
        rules[cols] = rules[cols].applymap(lambda x: tuple(x))
        for i in range(len(rules)):
            rule = rules.iloc[i]
            text += f"{rule['antecedents']} ==> {rule['consequents']}"
            text += f"(s: {rule['support']:.2f}, c: {rule['confidence']:.2f}, lift: {rule['lift']:.2f})\n"
        ax.text(0, 0, text)

        savefig(f'{self.output_path}/plot_top_rules.png')

        if show_figure:
            show()

    def analyse_per_metric(self, rules: DataFrame, metric: str, metric_values: list, show_figure=False) -> list:
        print(f'Analyse per {metric}...')
        conf = {'avg': [], 'top25%': [], 'top10': []}
        lift = {'avg': [], 'top25%': [], 'top10': []}
        top_conf = []
        top_lift = []
        nr_rules = []
        for m in metric_values:
            rs = rules[rules[metric] >= m]
            nr_rules.append(len(rs))
            conf['avg'].append(rs['confidence'].mean(axis=0))
            lift['avg'].append(rs['lift'].mean(axis=0))

            top_conf = rs.nlargest(int(0.25 * len(rs)), 'confidence')
            conf['top25%'].append(top_conf['confidence'].mean(axis=0))
            top_lift = rs.nlargest(int(0.25 * len(rs)), 'lift')
            lift['top25%'].append(top_lift['lift'].mean(axis=0))

            top_conf = rs.nlargest(10, 'confidence')
            conf['top10'].append(top_conf['confidence'].mean(axis=0))
            top_lift = rs.nlargest(10, 'lift')
            lift['top10'].append(top_lift['lift'].mean(axis=0))

        _, axs = subplots(1, 2, figsize=(10, 5), squeeze=False)
        multiple_line_chart(metric_values, conf, ax=axs[0, 0], title=f'Avg Confidence x {metric}',
                            xlabel=metric, ylabel='Avg confidence')
        multiple_line_chart(metric_values, lift, ax=axs[0, 1], title=f'Avg Lift x {metric}',
                            xlabel=metric, ylabel='Avg lift')
        if show_figure:
            show()

        self.plot_top_rules(top_conf, 'confidence', metric)
        self.plot_top_rules(top_lift, 'lift', metric)

        return nr_rules

    def quality_evaluation(self, show_figure=False):
        nr_rules_sp = self.analyse_per_metric(self.rules, 'support', self.var_min_sup)
        plot_line(self.var_min_sup, nr_rules_sp, title='Nr rules x Support', xlabel='support', ylabel='Nr. rules',
                  percentage=False)


    def quality_evaluation_per_confidence(self, show_figure=False):
        var_min_conf = [i * self.MIN_CONF for i in range(10, 5, -1)]
        nr_rules_cf = self.analyse_per_metric(self.rules, 'confidence', var_min_conf)
        plot_line(var_min_conf, nr_rules_cf, title='Nr Rules x Confidence', xlabel='confidence', ylabel='Nr Rules',
                  percentage=False)


if __name__ == "__main__":
    patternMiner = PatternMiner(data_set='nyc')

    patternMiner.show_patterns(show_figure=True)