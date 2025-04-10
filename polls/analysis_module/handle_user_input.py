import pandas as pd


def analysis_module(option, file):
    print(option)
    if file:
        df = pd.read_csv(file)
        print(df.head())
