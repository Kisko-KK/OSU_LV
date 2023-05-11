import pandas as pd
import numpy as np

"""
data = pd.read_csv("LV3\data_C02_emission.csv")

data.Make = data.Make.astype('category')
data.Model = data.Model.astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data.Transmission = data.Transmission.astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')


data = pd.read_csv("LV3\data_C02_emission.csv")
data = np.array(data)


data = data[data[:,11].argsort(kind='mergesort')]
print(np.mean(data[(data[:,0] == 'Audi') & (data[:,4] == 4),11]))
"""



