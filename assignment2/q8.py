import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def a8():
    df = pd.read_excel('LB_2.xlsx', sheet_name='thyroid0387_UCI')
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=["object"]).columns
    outlier_cols = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3-q1
        o = (df[col]<q1-1.5*iqr) | (df[col]>q3+1.5*iqr)
        if o.sum():
            outlier_cols.append(col)
    num_mn = list(set(num_cols)-set(outlier_cols))
    df[num_mn] = SimpleImputer(strategy='mean').fit_transform(df[num_mn])
    df[outlier_cols] = SimpleImputer(strategy='median').fit_transform(df[outlier_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    return df

if __name__ == '__main__':
    output_a8 = a8()
    print("A8 Output (First 5 rows after imputation):", output_a8.head())
