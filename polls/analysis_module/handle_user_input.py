import numpy as np
import pandas as pd
import statsmodels.api as sm


def analysis_module(option, file, features, target):
    df = pd.read_csv(file)
    result = {}
    data_input = features + [target]
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
        # TODO: Which feature should be returned?
        data = df[data_input].values.tolist()
        return data, percentage
    elif option == 'classification':
        class_counter = df[target].value_counts().to_dict()
        if len(class_counter) == 2:
            max_key = max(class_counter.values())
            min_key = min(class_counter.values())
            result['imbalance_ratio'] = max_key / min_key
        else:
            # TODO: imbalance degree
            print('multiclass classification')
        return result
