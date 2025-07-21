import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def a9():
    df = pd.read_excel('LB_2.xlsx', sheet_name='thyroid0387_UCI')
    num_cols = df.select_dtypes(include=[np.number]).columns
    normed = df.copy()
    normed[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return normed

if __name__ == '__main__':
    output_a9 = a9()
    print("A9 Output (First 5 rows of normalized data):", output_a9.head())
