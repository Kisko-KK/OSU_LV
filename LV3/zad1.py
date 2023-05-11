import pandas as pd

data = pd . read_csv('LV3\data_C02_emission.csv')


# a)
print(len(data))
print(data.info())
print(data.isnull().sum())
print(data.duplicated)


data.drop_duplicates()
data = data.dropna(axis = 0)
data = data.reset_index(drop = True)


data.Make = data.Make.astype('category')
data.Model = data.Model.astype('category')
data['Vehicle Class'] = data['Vehicle Class'].astype('category')
data.Transmission = data.Transmission.astype('category')
data['Fuel Type'] = data['Fuel Type'].astype('category')

print(data.info())


# b)
data.sort_values(by=['Fuel Consumption City (L/100km)'], inplace=True)
print(data['Fuel Consumption City (L/100km)'])
print(data.tail(3)[['Make', 'Model', 'Fuel Consumption City (L/100km)']])



# c)
print(data[(data['Engine Size (L)'] >= 2.5) & (
data['Engine Size (L)'] <= 3.5)]['CO2 Emissions (g/km)'])
print(data[(data['Engine Size (L)'] >= 2.5) & (
data['Engine Size (L)'] <= 3.5)]['CO2 Emissions (g/km)'].count())
print(data[(data['Engine Size (L)'] >= 2.5) & (
data['Engine Size (L)'] <= 3.5)]['CO2 Emissions (g/km)'].mean())


# d)
print(len(data[data.Make == 'Audi']))
print(data[(data.Make == 'Audi') & (data['Cylinders'] == 4)]
['CO2 Emissions (g/km)'].mean())


# e)
print(len(data[data.Cylinders == 4]))
print(len(data[data.Cylinders == 6]))
print(len(data[data.Cylinders == 8]))
print(data.Cylinders.corr(data['CO2 Emissions (g/km)']))


# f)
print(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].median())
print(data[data['Fuel Type'] == 'X']['Fuel Consumption City (L/100km)'].mean())

print(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].median())
print(data[data['Fuel Type'] == 'D']['Fuel Consumption City (L/100km)'].mean())


# g)
data[(data.Cylinders == 4) & (data['Fuel Type'] == 'D')].sort_values("Fuel Consumption City (L/100km)", ascending=False)
print(data.head(1))

# h)
print(len(data[data.Transmission.str.startswith('M')]))

# i)
print(data.corr(numeric_only=True))