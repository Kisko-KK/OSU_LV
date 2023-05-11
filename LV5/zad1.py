import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train)

# a)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(['red', 'blue']))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=ListedColormap(['red', 'blue']), marker='x')

plt.legend(['Podaci za učenje', 'Test podaci'])
plt.title('Podaci za učenje i testiranje')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

# b)
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)


# c)
coef = LogRegression_model.coef_
theta0 = LogRegression_model.intercept_

x1 = np.linspace(min(X_train[:, 0]), max(X_train[:, 0]), 100)
x2 = (-theta0 - coef[0][0]*x1) / coef[0][1]

plt.plot(x1, x2, label='Granica odluke')
plt.show()


# d)
y_test_p = LogRegression_model.predict(X_test)

cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()


print(f'Accuracy: {accuracy_score(y_test, y_test_p)}')
print(f'Precision: {precision_score(y_test, y_test_p)}')
print(f'Recall: {recall_score(y_test, y_test_p)}')


# e)
colors = ['black', 'green']
for i in range(len(y_test_p)):
    color = colors[int(y_test_p[i] == y_test[i])]
    plt.scatter(X_test[i, 0], X_test[i, 1], c=color, marker='o')
plt.show()