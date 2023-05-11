import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
# 1. Number of times pregnant
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 3. Diastolic blood pressure (mm Hg)
# 4. Triceps skin fold thickness (mm)
# 5. 2-Hour serum insulin (mu U/ml)
# 6. Body mass index (weight in kg/(height in m)^2)
# 7. Diabetes pedigree function
# 8. Age (years)
# 9. Class variable (0 or 1)
"""

data = np.loadtxt("exam\pima-indians-diabetes.csv",delimiter=",", dtype=float, skiprows=9)

data_df = pd.DataFrame(data)
data_df[8].replace({1 : 'has',
                    0 : 'hasnt'
                    }, inplace = True)
print(data_df)

print(f"Broj osoba: {data.shape[0]}")

data = data[data[:,5] != 0]
print(f"Broj osoba: {data.shape[0]}")

plt.scatter(data[:,7], data[:,5])
plt.xlabel("godine")
plt.ylabel("BMI")
plt.show()


#d)
print(np.min(data[:,5]))
print(np.max(data[:,5]))
print(np.mean(data[:,5]))

#e)
print(len(data[data[:,8]==1]))
print((np.min(data[data[:,8]==1,5])))
print((np.max(data[data[:,8]==1,5])))
print((np.mean(data[data[:,8]==1,5])))
