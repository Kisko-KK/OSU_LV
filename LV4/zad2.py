import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score

data = pd.read_csv('./LV3/data_C02_emission.csv')


ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
data["Fuel Type"] = X_encoded

X = data[["Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)","Engine Size (L)","Cylinders","Fuel Type"]].to_numpy()
y = data["CO2 Emissions (g/km)"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.coef_)


y_test_p = linearModel.predict(X_test_n)

plt.scatter(X_test_n[:,0],y_test, c='b', label='Real values', s = 1)
plt.scatter(X_test_n[:,0],y_test_p, c='r', label='Prediction', s = 1)
plt.legend()
plt.show()



print(f'Mean squared error: {mean_squared_error(y_test, y_test_p)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_test_p)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_test_p)}%')
print(f'R2 score: {r2_score(y_test, y_test_p)}')