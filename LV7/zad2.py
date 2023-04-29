import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs//test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255



# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))


# rezultatna slika
img_array_aprox = img_array.copy()

#prisutno razlicitih boja
print(len(np.unique(img_array,axis=0)))

#KMeans
km = KMeans(n_clusters=10, init = 'random', n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

centroids = km.cluster_centers_ * 255.0
colors = centroids[labels.astype(int)]
print(colors[0])

image = colors.reshape((w, h, d)).astype(int)

plt.figure()
plt.imshow(image)
plt.show()

plt.figure()
for i in range(1, 9):
    km = KMeans(n_clusters=i, init="random", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    plt.plot(i, km.inertia_, ".-r", linewidth=2)
    plt.xlabel("K")
    plt.ylabel("J")

plt.show()