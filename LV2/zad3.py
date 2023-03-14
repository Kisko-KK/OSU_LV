import numpy as np
import matplotlib.pyplot as plt



img = plt.imread("road.jpg")

img = img[:,:,0]



#a)
plt.imshow(img, cmap = "gray", alpha=0.4)
plt.show()

#b)
plt.imshow(img[:, round(len(img[:, 1]) * 0.25) : round(len(img[:, 1]) * 0.5)], cmap = "gray", alpha=0.4)
plt.show()


#c)
plt.imshow(np.rot90(np.rot90(np.rot90(img))), cmap = "gray")
plt.show()


#d)
plt.imshow(np.flip(img, axis=1), cmap = "gray")
plt.show()


