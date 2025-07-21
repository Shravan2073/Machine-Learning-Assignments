import pandas as pd
import numpy as np

def a4():
    df = pd.read_excel('LB_2.xlsx', sheet_name='thyroid0387_UCI')
    dtypes = df.dtypes
    num_cols = df.select_dtypes(include=[np.number]).columns
    min_max = df[num_cols].agg(['min', 'max'])
    missing = df.isnull().sum()
    outlier_counts = {}
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outliers = ((df[col] < (q1-1.5*iqr)) | (df[col] > (q3+1.5*iqr))).sum()
        outlier_counts[col] = outliers
    means = df[num_cols].mean()
    vars = df[num_cols].var()
    return dtypes, min_max, missing, outlier_counts, means, vars

if __name__ == '__main__':
    output_a4 = a4()
    print("A4 Output:", output_a4)
