import numpy as np
import pandas as pd
import statsmodels.api as sm

def handle_user_input(file):
    df = pd.read_csv(file)
    return df

def analysis_module(option, file, features, target):
    print(option)
    if file:
        if option == 'regression':
            df = pd.read_csv(file)
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
            data = df[['pH', 'total sulfur dioxide']].values.tolist()
            return data, percentage
        elif option == 'classification':
            print("classification")
            return 0
