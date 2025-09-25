import pandas as pd 
import numpy as np

path="DCT_mal.csv"
dat = pd.read_csv(path)

feature_vector = dat.iloc[:, :-1].values
class_labels = dat.iloc[: -1].values

d_class= np.unique(class_labels)

print('list of classe :')

print(len(d_class))

# try:
#     if len(d_class < 2 ):
#         for i in range(len(d_class)):
#             print(d_class[i])
#             print('niasdas')
#     else: 
#         asdasd
        

# except Exception as e:
#     print(f"size a lil too big da {e}")



