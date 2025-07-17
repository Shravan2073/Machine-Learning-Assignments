import pandas as pd
import numpy as np



df = pd.read_excel('C:\Users\shrav\Desktop\ML-sem-5\Machine-Learning-Assignments\assignment2\Lab Session Data.xlsx')

A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df['Payment (Rs)'].values

dimensionality = A.shape[1]  # number of columns/features
num_vectors   = A.shape[0]  # number of rows/data points



#read through the instructions again 