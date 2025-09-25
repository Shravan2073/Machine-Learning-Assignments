#Q1)
# import numpy as np 
# import pandas as pd 

# path = 'DCT_mal.csv'
# data = pd.read_csv(path)
# feature_vector = data.iloc[:,:-1].values
# class_lables = data.iloc[:,-1].values
# dist_class = np.unique(class_lables)

# print(dist_class)

# class_1 = dist_class[0]
# class_2 = dist_class[1]

# data_for_class1 = feature_vector[ class_1 == class_lables ] 
# print("data for c1" , data_for_class1, "len", len(data_for_class1))

# data_for_class2 = feature_vector[ class_2 == class_lables ] 

# mean_c1 = np.mean(data_for_class1, axis = 0 )
# mean_c2 = np.mean(data_for_class2, axis = 0 )

# std_c1= np.std(data_for_class1, axis= 0 )
# std_c2= np.std(data_for_class2, axis= 0 )

# cent_dist = np.linalg.norm(mean_c1 - mean_c2) 

# print(f"centroid of class {class_1} :\n{mean_c1}")
# print(f"centroid of class {class_2} :\n{mean_c2}")
# print(f"std dev of class {class_1}: \n {std_c1}")
# print(f"std dev of class {class_2}: \n {std_c2}")
# print('nigga')
# print(f"cent dist: {cent_dist}")

#Q2)

# import numpy as  np 
# import pandas as pd
# import matplotlib.pyplot as plt 

# dat = pd.read_csv('DCT_mal.csv', header = None) 
# req_feat = 0 
# feature_data = dat.iloc[:,req_feat].values

# mean_feat = np.mean(feature_data,axis=0)
# var_feat = np.var(feature_data, axis=0)

# hist_counts , hist_bins = np.histogram(feature_data, bins = 13 )
# print(f"bin count is : 13")
# print(hist_counts)
# print(hist_bins)

# plt.figure(figsize=(10,5))
# plt.hist(feature_data, bins= 13, color='skyblue', edgecolor='black')
# plt.title(f"hist for feature :{req_feat}")
# plt.xlabel(f"values")
# plt.ylabel(f"freaquency")
# plt.grid(True)
# plt.show()


#Q3)


# import numpy as np 
# import pandas as pd
# import matplotlib.pyplot as plt 

# dat= pd.read_csv('DCT_mal.csv', header = None)

# feat_1 = dat.iloc[0,:-1].values
# feat_2 = dat.iloc[1,:-1].values

# minsk = np.arange(1,11) 
# distances = []

# for i in minsk:
#     dist = np.sum(np.abs(feat_1 - feat_2) ** i ) ** (1/i)
#     distances.append(dist)

# plt.figure(figsize=(8,6))
# plt.plot(minsk,distances,marker = 'o', color = 'purple')
# plt.title("minsk dist for order r=0 to 10")
# plt.xlabel("minsk order")
# plt.ylabel("distance")
# plt.grid(True)
# plt.show()


#Q4 and Q5 

import numpy as np 
import pandas as pd 
from  sklearn.model_selection import train_test_split
from  sklearn.neighbors import KNeighborsClassifier 

dat = pd.read_csv('DCT_mal.csv')

feature_vect = dat.iloc[:,:-1].values 
class_lable = dat.iloc[:,-1].values

chosen_class = [3368, 3364]
selection_mask= np.isin(class_lable, chosen_class)

filter_feature= feature_vect[selection_mask]
filter_lable= class_lable[selection_mask]

xtrain,xtest,ytrain,ytest = train_test_split(filter_feature, filter_lable, test_size= 0.3, random_state=42)

 

n =10 
knn = KNeighborsClassifier(n)
knn.fit(xtrain,ytrain) 

score= knn.score(xtest,ytest)

predict = knn.predict(xtest)

sample_test = 

print(f"Test Set Accuracy (k=10): {score}")
print(f"this is the predicted val: {predict}")
print("Model trained successfully with k=10.")


    


