import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers



data = np.loadtxt("exam\pima-indians-diabetes.csv",delimiter=",", dtype=float, skiprows=9)

X = data[:,[0,1,2,3,4,5,6,7]]
y = data[:,8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

model = keras.Sequential ()
model.add( layers.Input ( shape =(8, )))
model.add(layers.Dense(12 , activation = "relu") )
model.add(layers.Dense(8 , activation = "relu") )
model.add(layers.Dense(1 , activation = "sigmoid") )
model.summary()

model.compile ( loss ="binary_crossentropy",
    optimizer ="adam",
    metrics = ["accuracy" ,])

history = model.fit ( X_train, y_train , batch_size = 7, epochs = 170, validation_split=0.15)

model.save("modelExercise.keras")

model = keras.models.load_model("modelExercise.keras")
model.summary()

predictions = model.predict(X_test)

score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


y_predictions = model.predict(X_test)
print(y_predictions)
y_predictions = np.around(y_predictions).astype(np.int32)
print(y_predictions)
cm = confusion_matrix(y_test, y_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

