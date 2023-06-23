import pandas as pd

def import_data():
    df = pd.read_csv('stacking_survival_analysis/data/data.csv')

    return df