import numpy as np
import matplotlib.pyplot as plt


x = [1, 3, 3, 2, 1]
y = [1, 1, 2, 2, 1]

plt.axis([0,4 , 0, 4])
plt.plot(x, y, linewidth = 2, marker = ".", markersize = 5, color = "red")

plt.show()