import pandas as pd
import numpy as np
data = pd.read_csv("data_C02_emission.csv")
# izdvajanje pojedinog stupca
print ( data ['Cylinders'])
print ( data . Cylinders )
# izdvajanje vise stupaca
print ( data [['Model','Cylinders']])
# izdvajanje redaka koristenjem iloc metode
print ( data . iloc [2:6 , 2:7])
print ( data . iloc [ :, 2:5])
print ( data . iloc [ :, [0 ,4 , 7] ])
