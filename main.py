import pandas as pd

data = pd.read_csv('data.csv')

x = data.drop('label', axis=1)
y = data['label']

