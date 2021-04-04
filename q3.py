import numpy as np
import matplotlib.pyplot as plt
import math

def gaus(A,b):
    # acessando as linhas
    for i in range(len(A)):
        # verificar qual o maior pivô
        pivo = math.fabs(A[i][i])
        linhaPivo = i
        for j in range(i+1, len(A)):
            if math.fabs(A[j][i]) > pivo:
                pivo = math.fabs(A[j][i])
                linhaPivo = j
        # permutação de linhas
        if linhaPivo != i:
            A[i], A[linhaPivo] = A[linhaPivo], A[i]
            b[i], b[linhaPivo] = b[linhaPivo], b[i]
        # eliminação de Gauss
        for m in range(i+1, len(A)):
            multiplicador = A[m][i]/A[i][i]
            for n in range(i, len(A)):
                A[m][n] -= multiplicador*A[i][n]
            b[m] -= multiplicador*b[i]
    print(A)
    print(b)

def svd(a, b):
    u, s, v = np.linalg.svd(a)
    c = np.dot(u.T, b)
    w = np.linalg.solve(np.diag(s), c)
    x = np.dot(v.T, w)
    return x

def gauss(A):
    n = len(A)
    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n + 1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    # Solve equation Ax=b for an upper triangular matrix A
    x = [0 for i in range(n)]
    for i in range(n - 1, -1, -1):
        x[i] = A[i][n] / A[i][i]
        for k in range(i - 1, -1, -1):
            A[k][n] -= A[k][i] * x[i]
    return x

def lu(A):
    n = len(A)
    for k in list(range(1, n, 1)):
        for i in list(range(k + 1, n + 1, 1)):
            m = A[i - 1][k - 1] / A[k - 1][k - 1]
            A[i - 1][k - 1] = m
            for j in list(range(k + 1, n + 1, 1)):
                A[i - 1][j - 1] = A[i - 1][j - 1] - m * A[k - 1][j - 1]
    return A

def lowerTriang(L, b):
    n = len(b)
    y = [0] * n
    for i in list(range(1, n + 1, 1)):
        s = 0
        for j in list(range(1, i, 1)):
            s = s + L[i - 1][j - 1] * y[j - 1]
        y[i - 1] = b[i - 1] - s
    return y

def upperTriang(U, b):
    n = len(b)
    x = [0] * n
    x[n - 1] = b[n - 1] / U[n - 1][n - 1]
    for i in list(range(n - 1, 0, -1)):
        s = 0
        for j in list(range(i + 1, n + 1, 1)):
            s = s + U[i - 1][j - 1] * x[j - 1]
        x[i - 1] = (b[i - 1] - s) / (U[i - 1][i - 1])
    return x

def gaussSeidel(A, b, vetorSolucao, N):
    it = 0
    while it < N:
        for i in range(len(A)):
            x = b[i]
            for j in range(len(A)):
                if i != j:
                    x -= A[i][j] * vetorSolucao[j]
            x /= A[i][i]
            vetorSolucao[i] = x
        it += 1
    return vetorSolucao

def jacobi(A, b, vetorSolucao, N):
    it = 0
    vetorAuxiliar = []
    for _ in range(len(vetorSolucao)):
        vetorAuxiliar.append(0)
    while it < N:
        for i in range(len(A)):
            x = b[i]
            for j in range(len(A)):
                if i != j:
                    x -= A[i][j]*vetorSolucao[j]
            x /= A[i][i]
            vetorAuxiliar[i] = x
        it += 1
        for p in range(len(vetorAuxiliar)):
            vetorSolucao[p] = vetorAuxiliar[p]
    return vetorSolucao


sist4_A = [[10, 2, 1.6, 3.5], [3, 13, -7, 1.4], [1, 2, 19, -4], [5, 1.5, 4.6, -21]]
sist4_b = [2, -7, -8, 5]
sist5_A = [[1, 2, 1.6, 3.5], [3, 5, -7, 1.4], [1, 2, 4, -4], [5, 1.5, 4.6, -1.2]]
sist5_b = [2, -7, -8, 5]
sist6_A = [
    [10, 20, 40, 70, 30, 20, 35],
    [3, 5, 8, 14, 17, 13, 22],
    [1, 2, 4, -4, -2, -1, -3],
    [1000, 2500, 4600, 3800, 7300, 8800, 9900],
    [54353, 23453, 42343, 34534, 45634, -23453, 74756],
    [0.01, 0.03, 0.045, -0.001, -0.5, -0.5, -0.25],
    [13, 23, 43, 73, -23, 22, -41]
]
sist6_b = [40, 13, 5, 750, 8000, 0.07, 42]

# QUESTÃO 3.1
print('SOLUÇÕES POR SVD')
print('Sistema 4', svd(sist4_A, sist4_b))
print('Sistema 5', svd(sist5_A, sist5_b))
print('Sistema 6', svd(sist6_A, sist6_b))
print('-' * 30)
# QUESTÃO 3.2
print('SOLUÇÕES SISTEMA 4')
print('Eliminação de Gauss:', gauss([sist4_A[i] + [sist4_b[i]] for i in range(4)]))

A = []
sol_y = []
sol_x = []
A.append(lu(sist4_A))
sol_y.append(lowerTriang(A[0], sist4_b))
sol_x.append(upperTriang(A[0], sol_y[0]))
print('Fatoração LU:', sol_x[0])

print('Gauss-Seidel:', gaussSeidel(sist4_A, sist4_b, [0, 0, 0, 0], 6))

print('Jacobi:', jacobi(sist4_A, sist4_b, [0, 0, 0, 0], 6), end = '\n\n')


print('SOLUÇÕES SISTEMA 5 (iteráveis não convergem!)')
print('Eliminação de Gauss:', gauss([sist5_A[i] + [sist5_b[i]] for i in range(4)]))

A = []
sol_y = []
sol_x = []
A.append(lu(sist5_A))
sol_y.append(lowerTriang(A[0], sist5_b))
sol_x.append(upperTriang(A[0], sol_y[0]))
print('Fatoração LU:', sol_x[0])

print('Gauss-Seidel:', gaussSeidel(sist5_A, sist5_b, [0, 0, 0, 0], 6))

print('Jacobi:', jacobi(sist5_A, sist5_b, [0, 0, 0, 0], 6))
print('-' * 30)

colors = 'g', 'y', 'r', 'b'
solucoes_GS = []
solucoes_jacobi = []
for i in range(6):
    solucoes_GS.append(gaussSeidel(sist4_A, sist4_b, [0, 0, 0, 0], i+1))
    solucoes_jacobi.append(jacobi(sist4_A, sist4_b, [0, 0, 0, 0], i+1))

for i in range(len(solucoes_GS)):
    plt.scatter([i+1]*4, solucoes_GS[i], c=colors)
plt.grid(True)
#plt.show()

for i in range(len(solucoes_jacobi)):
    plt.scatter([i+1]*4, solucoes_jacobi[i], c=colors)
plt.grid(True)
#plt.show()

# QUESTÃO 3.3
print('SOLUÇÕES SISTEMA 6')
print('Eliminação de Gauss:', gauss([sist6_A[i] + [sist6_b[i]] for i in range(7)]))

A = []
sol_y = []
sol_x = []
# A.append(lu(sist6_A))
# sol_y.append(lowerTriang(A[0], sist6_b))
# sol_x.append(upperTriang(A[0], sol_y[0]))
# print('Fatoração LU:', sol_x[0])

print('Gauss-Seidel:', gaussSeidel(sist6_A, sist6_b, [0, 0, 0, 0, 0, 0, 0], 6))

print('Jacobi:', jacobi(sist6_A, sist6_b, [0, 0, 0, 0, 0, 0, 0], 6))
print('-' * 30)