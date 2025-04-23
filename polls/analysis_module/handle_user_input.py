from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analysis_module(option, file, features, target):
    df = pd.read_csv(file)
    result = {}
    if option == 'regression':
        x = sm.add_constant(df[features])
        y = df[target]
        model = sm.OLS(y, x).fit()
        influence = model.get_influence()
        cooks_d, _ = influence.cooks_distance
        threshold = 4 / len(df)
        influential_points = np.where(cooks_d > threshold)[0]
        percentage = len(influential_points) / len(df) * 100
        data_input = features + [target]
        data = df[data_input].values.tolist()
        return data, percentage
    elif option == 'classification':
        class_counter = df[target].value_counts().to_dict()
        data = df[target].values.tolist()
        if len(class_counter) == 2:
            max_key = max(class_counter.values())
            min_key = min(class_counter.values())
            result['imbalance_ratio'] = max_key / min_key
        else:
            empirical_distribution = list(class_counter.values())
            mean = sum(empirical_distribution) / len(empirical_distribution)
            balanced_distribution = [mean] * len(empirical_distribution)
            minority_classes = [x for x in empirical_distribution if x != max(empirical_distribution)]
            num_minority_classes = len(minority_classes)
            max_diff_vals = []
            if num_minority_classes:
                max_diff_val = max(minority_classes, key=lambda x: abs(x - mean))
                max_diff_vals = [x for x in minority_classes if x == max_diff_val]
            numerator = sqrt(sum((x - y) ** 2 for x, y in zip(empirical_distribution, balanced_distribution)))
            denominator = sqrt(sum((x - y) ** 2 for x, y in zip(max_diff_vals, [mean] * len(max_diff_vals))))
            result['imbalance_degree'] = numerator / denominator + (num_minority_classes - 1) \
                if num_minority_classes else -1
        return data, result
    elif option == 'clustering':
        data = df[features].values.tolist()
        # TODO: clustering analysis
        return data, None
