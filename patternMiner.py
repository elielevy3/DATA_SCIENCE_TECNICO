from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, show, subplots
from ds_charts import dummify, plot_line, multiple_line_chart
from  mlxtend.frequent_patterns import apriori, association_rules
from data import Data

class PatternMiner():
    def __init__(self, data_set='air_quality'):
        data = Data()
        self.data_set = data_set
        self.data = data.air_quality_data if data_set == 'air_quality' else data.nyc_data
        self.data = self.data[:5000]

    def show_patterns(self, show_figure=False):
        MIN_SUP: float = 0.001
        var_min_sup =[0.2, 0.1] + [i*MIN_SUP for i  in range(100, 0, -10)]

        patterns: DataFrame = apriori(self.data, min_support=MIN_SUP, use_colnames=True, verbose=True)
        print(len(patterns),'patterns')
        nr_patterns = []
        for sup in var_min_sup:
            pat = patterns[patterns['support']>=sup]
            nr_patterns.append(len(pat))

        figure(figsize=(6, 4))
        plot_line(var_min_sup, nr_patterns, title='Nr Patterns x Support', xlabel='support', ylabel='Nr Patterns')

        if show_figure:
            show()

    def computeNumberOfRules(self):
        MIN_CONF: float = 0.1
        rules = association_rules(patterns, metric='confidence', min_threshold=MIN_CONF * 5, support_only=False)
        print(f'\tfound {len(rules)} rules')



if __name__ == "__main__":
    patternMiner = PatternMiner(data_set='air_quality')

    patternMiner.show_patterns(show_figure=True)