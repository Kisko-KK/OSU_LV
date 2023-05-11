import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data = pd.read_csv('./LV3/data_C02_emission.csv')


data = data.drop(['Make', 'Model'], axis=1)                     

input = [
    'Engine Size (L)', 
    'Cylinders', 
    'Fuel Consumption City (L/100km)', 
    'Fuel Consumption Hwy (L/100km)', 
    'Fuel Consumption Comb (L/100km)', 
    'Fuel Consumption Hwy (L/100km)', 
    'Fuel Type'
]

output = ['CO2 Emissions (g/km)']

ohe = OneHotEncoder()                                               #KATEGORIČKE velicine pretvara u NUMERICKE
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()
data['Fuel Type'] = X_encoded                                       #Zamjena podataka sa enkodiranim/transformiranim podatcima


X = data[input].to_numpy()                                          #VAŽNO iz dataframe-a u ndarray pretvori UVIJEK DODAT PRIJE train_test_split
y = data[output].to_numpy()

X_train , X_test , y_train , y_test = train_test_split (X , y , test_size = 0.2 , random_state =1 )

linearModel = lm.LinearRegression()
linearModel.fit( X_train, y_train )

print(linearModel.intercept_, linearModel.coef_)

y_test_p = linearModel.predict(X_test)                                 #uvijek prediktamo nakon što istreniramo model


plt.scatter(X_test[:, 0], y_test, c='blue', s=1)
plt.scatter(X_test[:, 0], y_test_p, c='red', s=1)
plt.xlabel(input[0])
plt.ylabel('CO2 Emissions (g/km)')
plt.legend(('Real output', 'Predicted output'))
plt.show()


error = abs(y_test_p - y_test)                                         #da greška ne bude - 
print(np.max(error))

max_error_id = np.argmax(error)                                        #izvlaci index di se nalazi maximalna pogreska
data = pd.read_csv('./LV3/data_C02_emission.csv')
max_error_model = data.iloc[max_error_id, 1]                           #izvukli red sa maximalnom greskom i izvukli prvu kolonu jer je to Model
print(max_error_model)

