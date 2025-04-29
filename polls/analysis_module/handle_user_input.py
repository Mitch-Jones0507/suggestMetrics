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
        is_polynomial = query.get("isPolynomial") == "true"
        degree_of_polynomial = query.get("degreeOfPolynomial")
        result = regression_analysis(df, features, target, is_polynomial, degree_of_polynomial)
        data_input = features + [target]
        data = df[data_input].values.tolist()
        return data, result
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
        imbalance_ratio = round(max_num / min_num, 2)
        if imbalance_ratio > 3.0 and class_counter[positive_class] == min_num:
            result['imbalance_ratio'] = (imbalance_ratio, 'Macro F1')
        else:
            result['imbalance_ratio'] = (imbalance_ratio, 'Accuracy')
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
        imbalance_degree = round(numerator / denominator + (num_minority_classes - 1),
                                 2) if num_minority_classes else -1
        if imbalance_degree <= 1.0:
            result['imbalance_degree'] = (imbalance_degree, 'Accuracy')
        else:
            result['imbalance_degree'] = (imbalance_degree, 'Macro F1')

        if num_minority_classes >= 3:
            result['num_minority_classes'] = (num_minority_classes, 'Macro F1')
        else:
            result['num_minority_classes'] = (num_minority_classes, 'Accuracy')

    if is_cost_sensitive:
        result["is_cost_sensitive"] = (is_cost_sensitive, 'Macro F1')
    else:
        result["is_cost_sensitive"] = (is_cost_sensitive, 'Accuracy')

    if is_specific:
        result["is_specific"] = (is_specific, 'Macro F1')
    else:
        result["is_specific"] = (is_specific, 'Accuracy')

    return result


def regression_analysis(df, features, target, is_polynomial, degree_of_polynomial):
    result = {}

    x = sm.add_constant(df[features])
    y = df[target]
    model = sm.OLS(y, x).fit()
    influence = model.get_influence()
    cooks_d, _ = influence.cooks_distance
    threshold = 4 / len(df)
    influential_points = np.where(cooks_d > threshold)[0]
    outlier_rate = round(len(influential_points) / len(df), 2)
    if outlier_rate < 0.05:
        result["outlier_rate"] = (outlier_rate * 100, 'R Square')
    else:
        result["outlier_rate"] = (outlier_rate * 100, 'MAE')

    if len(features) < 2:
        mean = np.mean(df[target])
        std = np.std(df[target], ddof=0)
        coefficient_of_variation = round(std / mean, 2)
        if coefficient_of_variation > 0.1:
            result["coefficient_of_variation"] = (coefficient_of_variation, 'R Square')
        else:
            result["coefficient_of_variation"] = (coefficient_of_variation, 'MAE')
    else:
        condition_number = round(np.linalg.cond(df[features].to_numpy()), 2)
        if condition_number < 30:
            result["condition_number"] = (condition_number, 'R Square')
        else:
            result["condition_number"] = (condition_number, 'MAE')

        subjects_per_predictor = round(len(df) / len(features), 2)
        if subjects_per_predictor > 10:
            result["subjects_per_predictor"] = (subjects_per_predictor, 'R Square')
        else:
            result["subjects_per_predictor"] = (subjects_per_predictor, 'MAE')

    if is_polynomial:
        if degree_of_polynomial == 'more than 3':
            result["degree_of_polynomial"] = (degree_of_polynomial, 'MAE')
        else:
            result["degree_of_polynomial"] = (degree_of_polynomial, 'R Square')
    return result
