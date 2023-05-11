import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# ucitaj podatke
data = pd.read_csv("LV6\Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()


#ZADATAK 3

SVM_model = svm.SVC(kernel ='rbf', gamma = 1 , C=1 )
SVM_model.fit( X_train_n , y_train )

y_train_p = SVM_model.predict(X_train_n)
y_test_p = SVM_model.predict(X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()




#ZADATAK 4



svm_model2 = svm.SVC(kernel="rbf", gamma=1, C=10)
svm_model2.fit( X_train_n , y_train )                                                   #izgradi model SVC


param_grid = {"C": [10, 100, 100], "gamma": [10, 1, 0.1, 0.01]}
svm_gs = GridSearchCV(svm_model2, param_grid, cv=5)                                     #Pomocu unakrsne validacije nadi parametre
svm_gs.fit(X_train_n, y_train)

print(svm_gs.best_params_, svm_gs.best_score_)

y_train_p = svm_model2.predict(X_train_n)
y_test_p = svm_model2.predict(X_test_n)

# granica odluke pomocu SVM
plot_decision_regions(X_train_n, y_train, classifier=svm_model2)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()