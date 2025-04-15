import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np

def analysis_module(option, file):
    print(option)
    if file:
        if option == 'regression':
            df = pd.read_csv(file)
            print(df.head())
            x = sm.add_constant(df[['pH']])
            y = df['residual sugar']
            model = sm.OLS(y, x).fit()
            influence = model.get_influence()
            cooks_d, _ = influence.cooks_distance
            threshold = 4 / len(df)
            influential_points = np.where(cooks_d > threshold)[0]
            print("Influential points (potential outliers):", influential_points)
            percentage = len(influential_points) / len(df) * 100
            print("Percentage of influential points:", percentage)
            data = df[['pH', 'total sulfur dioxide']].values.tolist()
            return data, percentage
        elif option == 'classification':
            print("classification")
            return 0
