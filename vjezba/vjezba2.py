from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('exam/titanic.csv')

#1.zadatak
##################################################################################
"""
#PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
print(len(data[data['Survived'] == 1]))

survived = data[data['Survived'] == 1]

num_of_men = len(data[data['Sex'] == 'male'])
num_of_women = len(data[data['Sex'] == 'female'])
num_of_people = data.shape[0]

sex = ['male', 'female']
means = [(num_of_men/num_of_people)*100,(num_of_women/num_of_people)*100]
plt.bar(sex, means)
plt.xlabel("Sex")
plt.ylabel("Broj osoba")
plt.title("bla")
plt.show()

data.dropna(inplace=True)

print(survived[survived["Sex"] == "male"]["Age"])
"""
#################################################################################


#2.zadatak
data = data.dropna(axis = 0)
data = data.reset_index(drop = True)

X = data[["Pclass", "Sex", "Fare", "Embarked"]]
y = data["Survived"]

X['Sex'].replace({'male' : 0,
                        'female' : 1
                        }, inplace = True)
X['Embarked'].replace({'C' : 0,
                        'Q' : 1,
                        'S' : 2
                        }, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

print(X)
sc = MinMaxScaler()                             
X_train_n = sc.fit_transform( X_train )            
X_test_n = sc.transform( X_test )

#a)
KNN_model = KNeighborsClassifier( n_neighbors = 29 )                                         #Izgradi KNN model
KNN_model.fit(X_train_n , y_train)

plt.title("Dead depending on sex and fare")
plt.xlabel("Fare")
plt.ylabel("Sex")
plt.scatter(X_train['Fare'], X_train['Sex'], c=y_train)
plt.show()

y_test_p_ = KNN_model.predict( X_test_n )
y_train_p_ = KNN_model.predict( X_train_n )

print("KNN model1: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_))))


param_grid = {'n_neighbors': np.arange(1,100)}
knn_gscv = GridSearchCV(KNN_model, param_grid, cv = 5)                                    #Pomocu unakrsne validacije odredite optimalnu vrijednost K
knn_gscv.fit( X_train_n , y_train )


print ( knn_gscv.best_score_ , knn_gscv.best_params_)