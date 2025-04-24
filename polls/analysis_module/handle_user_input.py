from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm


def analysis_module(query, file):
    option = query.get("task")
    df = pd.read_csv(file)
    if option == 'regression':
        features = query.getlist("features")
        target = query.get("target")
        percentage = regression_analysis(df, features, target)
        data_input = features + [target]
        data = df[data_input].values.tolist()
        return data, percentage
    elif option == 'classification':
        target = query.get("target")
        positive_class = query.get("positiveClass")
        is_cost_sensitive = query.get("isCostSensitive") == "true"
        is_specific = query.get("isSpecific") == "true"
        result = classification_analysis(df, target, positive_class, is_cost_sensitive, is_specific)
        data = df[target].values.tolist()
        return data, result
    elif option == 'clustering':
        features = query.getlist("features")
        data = df[features].values.tolist()
        # TODO: clustering analysis
        return data, None


def classification_analysis(df, target, positive_class, is_cost_sensitive, is_specific):
    result = {}

    class_counter = df[target].value_counts().to_dict()
    if len(class_counter) == 2:
        max_num = max(class_counter.values())
        min_num = min(class_counter.values())
        imbalance_ratio = max_num / min_num
        if imbalance_ratio > 3.0 and class_counter[positive_class] == min_num:
            result['imbalance_ratio'] = (imbalance_ratio, 'macF1')
        else:
            result['imbalance_ratio'] = (imbalance_ratio, 'accuracy')
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
        imbalance_degree = numerator / denominator + (num_minority_classes - 1) if num_minority_classes else -1
        if imbalance_degree <= 1.0:
            result['imbalance_degree'] = (imbalance_degree, 'accuracy')
        else:
            result['imbalance_degree'] = (imbalance_degree, 'macF1')

        if num_minority_classes >= 3:
            result['num_minority_classes'] = (num_minority_classes, 'macF1')
        else:
            result['num_minority_classes'] = (num_minority_classes, 'accuracy')

    if is_cost_sensitive:
        result["is_cost_sensitive"] = (is_cost_sensitive, 'macF1')
    else:
        result["is_cost_sensitive"] = (is_cost_sensitive, 'accuracy')

    if is_specific:
        result["is_specific"] = (is_specific, 'macF1')
    else:
        result["is_specific"] = (is_specific, 'accuracy')

    return result


def regression_analysis(df, features, target):
    x = sm.add_constant(df[features])
    y = df[target]
    model = sm.OLS(y, x).fit()
    influence = model.get_influence()
    cooks_d, _ = influence.cooks_distance
    threshold = 4 / len(df)
    influential_points = np.where(cooks_d > threshold)[0]
    percentage = len(influential_points) / len(df) * 100
    return percentage
