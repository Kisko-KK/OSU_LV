from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris = datasets.load_iris()

print(iris)

target = iris['target']
data = iris['data']


types = iris['target_names']
setosa = data[target==0]
versicolour = data[target==1]
virginica = data[target==2]


plt.scatter(versicolour[:,0], versicolour[:,2], c= "blue")
plt.scatter(virginica[:,0], virginica[:,2], c= "red")
plt.xlabel("Duljina")
plt.ylabel('sirina')
plt.legend(('Versi', 'Virgi'))
plt.show()

setosa_mean = np.mean(setosa[:,3])
versi_mean = np.mean(versicolour[:,3])
virgi_mean = np.mean(virginica[:,3])

plt.bar( types,[setosa_mean, versi_mean, virgi_mean], color = "red")
plt.show()


print(len(virginica[virginica[:,3] > virgi_mean] ))



for i in range(1, 9):
    km = KMeans(n_clusters=i, init="random", n_init=5, random_state=0)
    km.fit(data)
    plt.plot(i, km.inertia_, ".-r", linewidth=2)                                            #J parametar
    plt.xlabel("K")
    plt.ylabel("J")

plt.show()

km = KMeans(n_clusters=3, init="random", n_init=5, random_state=0)
km.fit(data)
labels = km.predict(data)

plt.scatter(data[:,0], data[:,1],c = labels, s=2, label = types)

plt.scatter(km.cluster_centers_[0,0],km.cluster_centers_[0,1], c = "red", label = "Centroid 1")
plt.scatter(km.cluster_centers_[1,0],km.cluster_centers_[1,1], c = "red")
plt.scatter(km.cluster_centers_[2,0],km.cluster_centers_[2,1], c = "red")
plt.legend()
plt.show()


total = len(target)
print(target)
print(labels)

good = np.count_nonzero(np.where(labels == target))

print(f"Total: {total}")
print(f"Good: {good}")
print(f"%: {good/total}")