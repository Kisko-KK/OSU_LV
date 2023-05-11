from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


iris = datasets.load_iris()

target = iris['target']
data = iris['data']

X = data
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, stratify=y, random_state = 10)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

model = keras.Sequential ()
model.add( layers.Input ( shape =(4, )))
model.add(layers.Dense(10 , activation = "relu") )
model.add(layers.Dropout(rate=0.3))   
model.add(layers.Dense(7 , activation = "relu") )
model.add(layers.Dropout(rate=0.3))   
model.add(layers.Dense(5 , activation = "relu") )
model.add(layers.Dense(3 , activation = "softmax") )
model.summary()
"""
model.compile ( loss ="categorical_crossentropy",
    optimizer ="adam",
    metrics = ["accuracy" ,])

history = model.fit ( X_train_n, y_train , batch_size = 7, epochs = 500 , validation_split = 0.1)

model.save("modell.keras")
"""
model = keras.models.load_model("modell.keras")
model.summary()


score = model.evaluate(X_test_n, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


predictions = model.predict(X_test_n)
print(predictions)

y_pred = np.argmax(predictions, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()