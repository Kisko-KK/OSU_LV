import numpy as np
import matplotlib.pyplot as plt


black = np.ones((50,50))
white = np.zeros((50,50))

left = np.vstack((black, white))
right = np.vstack((white, black))


img = np.hstack((left, right))

plt.imshow(img, cmap="gray")
plt.show()