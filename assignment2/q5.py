import pandas as pd
import numpy as np

def a5():
    df = pd.read_excel('LB_2.xlsx', sheet_name='thyroid0387_UCI')
    # Only columns with just 't' and 'f' (no NaN or '?')
    def is_binary(col):
        vals = set(df[col].dropna().unique())
        vals = set([str(x).lower() for x in vals if x != '?'])
        return vals <= {'t','f'}
    bin_cols = [col for col in df.columns if is_binary(col)]
    mapping = {'f': 0, 't': 1}
    v0 = df.iloc[0][bin_cols].map(mapping).astype(int).fillna(0).values
    v1 = df.iloc[1][bin_cols].map(mapping).astype(int).fillna(0).values
    f11 = np.sum((v0 == 1) & (v1 == 1))
    f10 = np.sum((v0 == 1) & (v1 == 0))
    f01 = np.sum((v0 == 0) & (v1 == 1))
    f00 = np.sum((v0 == 0) & (v1 == 0))
    jc = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) else 0
    smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) else 0
    return jc, smc

if __name__ == "__main__":
    jc, smc = a5()
    print("A5 Output (JC, SMC):", jc, smc)
