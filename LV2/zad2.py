import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("LV2\data.csv",delimiter=",", dtype=float, skiprows=1)

#data = data[~np.isnan(data[:,2])]
#print(np.isnan(data[:,2]))

#a)
print(data.shape[0])

#b)
height = data[:,1]
weight = data[:,2]

plt.scatter(height, weight)
plt.xlabel("visina")
plt.ylabel("težina")
plt.title("Odnos visine i težine osoba")
plt.show()


#c)
data50 = data[::50, :]
plt.figure()
height = data50[:,1]
weight = data50[:,2]

plt.scatter(height, weight)
plt.xlabel("visina")
plt.ylabel("težina")
plt.title("Odnos visine i težine svakih 50. osoba")
plt.show()

#d)
print(np.max(height))
print(np.min(height))
print(np.mean(height))

#e)
women = data[data[:,0] == 0,:]
print(women)
heightWomen = women[:,1]

print(np.max(heightWomen))
print(np.min(heightWomen))
print(np.mean(heightWomen))

men = data[data[:,0] == 1,:]
heightMen = men[:,1]

print(np.max(heightMen))
print(np.min(heightMen))
print(np.mean(heightMen))
