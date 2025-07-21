import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def a7():
    df = pd.read_excel('LB_2.xlsx', sheet_name='thyroid0387_UCI')
    # Binary columns as above
    def is_binary(col):
        vals = set(df[col].dropna().unique())
        vals = set([str(x).lower() for x in vals if x != '?'])
        return vals <= {'t','f'}
    bin_cols = [col for col in df.columns if is_binary(col)]
    mapping = {'f': 0, 't': 1}
    binmat = df.iloc[:20][bin_cols].applymap(mapping.get).astype(int).fillna(0).values
    
    n = binmat.shape[0]
    jc_mat = np.zeros((n,n))
    smc_mat = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a = binmat[i]
            b = binmat[j]
            f11 = np.sum((a==1)&(b==1))
            f10 = np.sum((a==1)&(b==0))
            f01 = np.sum((a==0)&(b==1))
            f00 = np.sum((a==0)&(b==0))
            denom_jc = (f11+f10+f01)
            jc_mat[i,j] = f11 / denom_jc if denom_jc else 0
            denom_smc = (f11+f10+f01+f00)
            smc_mat[i,j] = (f11+f00)/denom_smc if denom_smc else 0
    # Cosine on NUMERIC columns
    num_cols = df.select_dtypes(include=np.number).columns
    nummat = df.iloc[:20][num_cols].fillna(0).values
    norm = np.linalg.norm(nummat, axis=1, keepdims=True)
    cos_mat = nummat @ nummat.T / (norm @ norm.T)
    sns.heatmap(jc_mat)
    plt.title("A7 JC Heatmap")
    plt.show()
    sns.heatmap(smc_mat)
    plt.title("A7 SMC Heatmap")
    plt.show()
    sns.heatmap(cos_mat)
    plt.title("A7 Cosine Heatmap")
    plt.show()
    return jc_mat, smc_mat, cos_mat

if __name__ == "__main__":
    mats = a7()
    print("A7 Output (JC, SMC, Cosine matrices):", [m.shape for m in mats])
