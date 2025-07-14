def multiply(a, b):
    n = len(a)
    result = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result

def matrixpower(matrix, m):
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    base = [row[:] for row in matrix]
    
    while m > 0:
        if m % 2 == 1:
            result = multiply(result, base)
        base = multiply(base, base)
        m //= 2
    
    return result

matrix_a = [[1, 2], [3, 4]]
power = 3
result_matrix = matrixpower(matrix_a, power)
print(f"Matrix A^{power}:")
for row in result_matrix:
    print(row)