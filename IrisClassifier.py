#%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

#iris_data = pd.read_csv('E:\Data Science\Iris Flower\iris-data.csv')
iris_data = pd.read_csv('E:\Data Science\Iris Flower\iris-data.csv', na_values=['NA'])
iris_data.head()
iris_data.describe()

iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'
iris_data.loc[iris_data['class'] == 'Iris-setossa', 'class'] = 'Iris-setosa'

iris_data['class'].unique()

sb.pairplot(iris_data.dropna(), hue='class')