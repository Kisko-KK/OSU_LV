import numpy as np
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

model = keras.models.load_model("FCN/model.keras")

# skaliranje slike na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = x_train_s.reshape((60000, 784))
x_test_s = x_test_s.reshape((10000, 784))

y_pred = model.predict(x_test_s)
y_pred_labels = np.argmax(y_pred, axis=1)

misclassified_idx = np.where(y_pred_labels != y_test)[0]


for i in range(0,5):
    idx = misclassified_idx[i]
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"True: {y_test[idx]} Pred: {y_pred_labels[idx]}")
    plt.show()