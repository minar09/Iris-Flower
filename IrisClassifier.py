import pandas as pd

#iris_data = pd.read_csv('iris-data.csv')
iris_data = pd.read_csv('iris-data.csv', na_values=['NA'])
iris_data.head()
