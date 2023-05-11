import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


data = pd.read_csv('./LV3/data_C02_emission.csv')

#a)
input_variables = [                                     
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)',
    'Engine Size (L)',
    'Cylinders']

output_variable = ['CO2 Emissions (g/km)']              

X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state =1 )

#b) Ovisnost CO2 plinova o nekoj numerickoj velicini
plt.scatter(x=X_train[:, 0], y=y_train, c="b")        
plt.scatter(x=X_test[:, 0], y=y_test, c="r")
plt.show()

#c) 
sc = MinMaxScaler()                                #STANDARDIZACIJA VELICINA
X_train_n = sc.fit_transform( X_train )            #namjesti veličine i skalira da bi bilo izmedu 0 i 1 
X_test_n = sc.transform( X_test )                  #transformira veličine tj. skalira i testne podatke kako bi bili kompatibil i sa onima iz train skupa

plt.subplot(2, 1, 1)
plt.hist(X_train[:, 0])
plt.title('Before Scaling')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.hist(X_train_n[:, 0])
plt.title('After Scaling')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')
plt.show()


#d) Izgradi Linearni regresijski model

linearModel=lm.LinearRegression()
linearModel.fit(X_train_n,y_train)
print(linearModel.coef_)


#e) 
y_test_p=linearModel.predict(X_test_n)

plt.scatter(X_test_n[:,0],y_test, c='b', label='Real values', s = 1)
plt.scatter(X_test_n[:,0],y_test_p, c='r', label='Prediction', s = 1)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#f)

print(f'Mean squared error: {mean_squared_error(y_test, y_test_p)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_test_p)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_test_p)}%')
print(f'R2 score: {r2_score(y_test, y_test_p)}')
