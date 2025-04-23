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
        print("Influential points (potential outliers):", influential_points)
        percentage = len(influential_points) / len(df) * 100
        print("Percentage of influential points:", percentage)
        data_input = features + [target]
        data = df[data_input].values.tolist()
        return data, percentage
    elif option == 'classification':
        class_counter = df[target].value_counts().to_dict()
        data = df[target].values.tolist()
        print(class_counter)
        if len(class_counter) == 2:
            max_key = max(class_counter.values())
            min_key = min(class_counter.values())
            result['imbalance_ratio'] = max_key / min_key
            print(result['imbalance_ratio'])
        else:
            empirical_distribution = list(class_counter.values())
            mean = sum(empirical_distribution) / len(empirical_distribution)
            balanced_distribution = [mean] * len(empirical_distribution)
            minority_classes = [x for x in empirical_distribution if x != max(empirical_distribution)]
            num_minority_classes = len(minority_classes)
            max_diff_val = max(minority_classes, key=lambda x: abs(x - mean)) if num_minority_classes else 0
            numerator = sqrt(sum((x - y) ** 2 for x, y in zip(empirical_distribution, balanced_distribution)))
            # TODO
            denominator = sqrt((max_diff_val - mean) ** 2)
            result['imbalance_degree'] = numerator / denominator * (num_minority_classes - 1) if max_diff_val != 0 else -1
        return data, result
    elif option == 'clustering':
        data = df[features].values.tolist()
        # TODO: clustering analysis
        return data, None
