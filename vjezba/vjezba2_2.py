import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv('exam/titanic.csv')

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


sc = MinMaxScaler()                             
X_train_n = sc.fit_transform( X_train )            
X_test_n = sc.transform( X_test )

model = keras.Sequential()
model.add(layers.Input(shape = (4, )))
model.add(layers.Dense(16, activation = "relu"))
model.add(layers.Dense(8, activation = "relu"))
model.add(layers.Dense(4, activation = "relu"))
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()

#b)
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy", ])
#c)
batch_size = 5
epochs = 100
history = model.fit(X_train_n , y_train, batch_size = batch_size, epochs = epochs, validation_split = 0.1)
predictions = model.predict(X_test)
score = model.evaluate(X_test, y_test ,verbose = 0)
print(score)
#d)
model.save("FCNispit1/")
del model
#e)
model = load_model("FCNispit1/")
model.summary ()

score = model.evaluate(X_test, y_test, verbose = 0)
print("Evaluate model: ", score)
#f)
predictions = model.predict(X_test)
predictions = np.round(predictions)
cm = confusion_matrix(y_test , predictions)
print ("Confusion matrix : " , cm )
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , predictions))
disp.plot()
plt.show()

