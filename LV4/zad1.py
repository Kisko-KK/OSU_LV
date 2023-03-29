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


#a)
data = pd.read_csv('./LV3/data_C02_emission.csv')
data=data.drop(["Make","Model"],axis=1)
x=data[["Engine Size (L)","Cylinders","Fuel Type","Fuel Consumption City (L/100km)","Fuel Consumption Hwy (L/100km)","Fuel Consumption Comb (L/100km)","Fuel Consumption Comb (mpg)"]]
y=data["CO2 Emissions (g/km)"].to_numpy()
ohe=OneHotEncoder()
X_encoded=pd.DataFrame(ohe.fit_transform(data[["Fuel Type"]]).toarray())

x = pd.concat([x.drop("Fuel Type", axis=1),X_encoded],axis=1).to_numpy()
print(x)

X_train , X_test , y_train , y_test = train_test_split (x, y, test_size = 0.2, random_state =1)



#b)
plt.scatter(X_train[:,5],y_train,color="blue", label="Training data")
plt.scatter(X_test[:,5],y_test,color="red", label="Testing data")
plt.show()


#c)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

plt.subplot(2, 1, 1)
plt.hist(X_train[:, 0])
plt.title('Before Scaling')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')

plt.subplot(2, 1, 2)
plt.hist(X_train_sc[:, 0])
plt.title('After Scaling')
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('Frequency')
plt.show()


#d)
linearModel=lm.LinearRegression()
linearModel.fit(X_train_sc,y_train)
print(linearModel.coef_)

#e)
y_test_p=linearModel.predict(X_test_sc)

plt.scatter(X_test_sc[:,0],y_test, c='b', label='Real values', s = 1)
plt.scatter(X_test_sc[:,0],y_test_p, c='r', label='Prediction', s = 1)
plt.xlabel('Fuel Consumption City (L/100km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.legend()
plt.show()

#f)

print(f'Mean squared error: {mean_squared_error(y_test, y_test_p)}')
print(f'Mean absolute error: {mean_absolute_error(y_test, y_test_p)}')
print(f'Mean absolute percentage error: {mean_absolute_percentage_error(y_test, y_test_p)}%')
print(f'R2 score: {r2_score(y_test, y_test_p)}')
