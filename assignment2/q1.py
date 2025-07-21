import pandas as pd
import numpy as np

def a1():
    df = pd.read_excel('LB_2.xlsx', sheet_name='Purchase data')
    df_A = df[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]]
    df_C = df[["Payment (Rs)"]]
    A = df_A.to_numpy()
    C = df_C.to_numpy()
    dim = A.shape[1]
    n_vectors = A.shape[0]
    rank = np.linalg.matrix_rank(A)
    X = np.linalg.pinv(A) @ C
    return dim, n_vectors, rank, X

if __name__ == '__main__':
    output_a1 = a1()
    print("A1 Output:", output_a1)
