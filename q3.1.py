################ Questão 3 ###########################

import numpy as np

######################## 3.1 ############################
print('Questão 3.1')
print('')
################## Matrizes ###########################

a = np.array([[10,  2,    1.6,  3.5],
              [3,   13,   -7,   1.4],
              [1,   2,    19,   -4],
              [5,   1.5,  4.6,  -21]],
              dtype=float)

b = np.array([[1, 2,    1.6,  3.5],
              [3, 5,    -7,   1.4],
              [1, 2,    4,    -4],
              [5, 1.5,  4.6,  -1.2]],
              dtype=float)

c = np.array([
    [10,    20,     40,     70,     30,     20,     35],
    [3,     5,      8,      14,     17,     13,     22],
    [1,     2,      4,      -4,     -2,     -1,     -3],
    [1000,  2500,   4600,   3800,   7300,   8800,   9900],
    [54353, 23453,  42343,  34534,  45634,  -23453, 74756],
    [0.01,  0.03,   0.045,  -0.001, -0.50,  -0.5,   -0.25],
    [13,    23,     43,     73,     -23,    22,     -41]], dtype=float)

l = [a,b,c]

y0 = np.array([[2], [-7], [-8], [5]], dtype=float)

y1 = np.array([[2], [-7], [-8], [5]], dtype=float)

y2 = np.array([[40], [13], [5], [750], [8000], [0.07], [42]], dtype=float)

Y = [y0, y1, y2]


################################### SVD ########################################

VH = []
U = []
S = []

for i in l:
    u, s, vh = np.linalg.svd(i, full_matrices=True)
    U.append(u)
    S.append(s)
    VH.append(vh)

################## Diagonalização das Matrizes 's' #############################

S_diagonal = []

for i in S:
    S_diagonal.append(np.diag(i))

################## Cálculo do Número de Condicionamento ########################
n = []
for i in S_diagonal:
    n.append(np.linalg.cond(i))

numero_equações = ['Primeiro', 'Segundo', 'Terceiro']
print('Número de Condicionamento')
for i, j in zip(n,numero_equações):
    print('O número de condicionamento do ',j,' sistema linear é: ',i)
print()
print('-------------------------------------------------------')

################### Resolução dos Sistemas Lineares ############################

def svdsolve(a,b):
    import numpy as np
    u,s,v = np.linalg.svd(a)
    c = np.dot(u.T,b)
    w = np.linalg.solve(np.diag(s),c)
    x = np.dot(v.T,w)
    return x


print('Solução do Sistema Linear por SDV')
print()
for i, j in zip(range(3),numero_equações):
    print('A solução do ',j,' sistema linear é: ')
    print(svdsolve(l[i],Y[i]))
    print()
print('-------------------------------------------------------')


#################################   3.2   ######################################

print('Questão 3.2')
print('')


############### Método de Eliminação por Gauss com Pivoteamento ################


def gauss(A, b):
    A = np.column_stack((A, b)).astype(float)  # matriz ampliada
    N = len(A)
    for i in range(N):
        k = np.argmax(np.abs(A[i:, i]))
        if k > 0:
            A[i + k], A[i] = np.copy(A[i]), np.copy(A[i + k])
        m = A[i, i:] / A[i, i]
        for j in range(i + 1, N):
            A[j, i:] -= A[j, i] * m
            A[j, :i] = 0.
    for i in range(N - 1, -1, -1):
        for j in range(N - 1, i, -1):
            A[i, -1] -= A[i, j] * A[j, -1]
        A[i, -1] /= A[i, i]
    return A[:, -1]


print('Método Eliminação de Gauss')
print()
for i, j in zip(range(2), numero_equações):
    print('A solução do ', j, ' sistema linear é: ')
    print(gauss(l[i], Y[i]))
    print()
print('-------------------------------------------------------')


######################### Método Fatoração LU ##################################

def LU(A):
    n = len(A)
    x = [0] * n
    for k in list(range(1, n, 1)):
        for i in list(range(k + 1, n + 1, 1)):
            m = A[i - 1][k - 1] / A[k - 1][k - 1]
            A[i - 1][k - 1] = m
            for j in list(range(k + 1, n + 1, 1)):
                A[i - 1][j - 1] = A[i - 1][j - 1] - m * A[k - 1][j - 1]
    return A


def solveLowerTriangular(L, b):
    n = len(b)
    y = [0] * n
    for i in list(range(1, n + 1, 1)):
        s = 0
        for j in list(range(1, i, 1)):
            s = s + L[i - 1][j - 1] * y[j - 1]
        y[i - 1] = b[i - 1] - s
    return y


def solveUpperTriangular(U, b):
    n = len(b)
    x = [0] * n
    x[n - 1] = b[n - 1] / U[n - 1][n - 1]
    for i in list(range(n - 1, 0, -1)):
        s = 0
        for j in list(range(i + 1, n + 1, 1)):
            s = s + U[i - 1][j - 1] * x[j - 1]
        x[i - 1] = (b[i - 1] - s) / (U[i - 1][i - 1])
    return x


A = []
solução_y = []
solução_x = []

for i in range(2):
    A.append(LU(l[i]))
    solução_y.append(solveLowerTriangular(A[i], Y[i]))
    solução_x.append(solveUpperTriangular(A[i], solução_y[i]))

print('Método Fatoração LU')
print()
for i, j in zip(range(2), numero_equações):
    print('A solução do ', j, ' sistema linear é: ')
    print(np.transpose(np.transpose(solução_x[i])))
    print()
print('-------------------------------------------------------')


##################### Método Iterativo Gauss Seidel  ###########################

def gauss_Seidel(A, b, X):
    x = X
    n = np.size(x, 1)
    nV = math.inf
    tolerancia = 1e-5
    itr = 0
    iteração = 1
    while abs(nV) > tolerancia:
        x_antigo = x
        for i in range(1, n):
            sigma = 0;
            for j in range(1, (i - 1)):
                sigma += A[i, j] * x[j]
            for k in range((i + 1), n):
                sigma += A[i, j] * x_antigo[j]
            x.append(((1 / A[i, i]) * (b[i] - sigma)))
            iteração += 1
        itr += 1
        nV = np.linalg.norm(x_antigo - x)
        print(x)


gauss_Seidel(l[0], Y[0], [[0], [0], [0], [0]])

#################################   3.3   ######################################
print('Questão 3.3')
print('')


def gauss_3_3(A, b):
    A = np.column_stack((A, b)).astype(float)  # matriz ampliada
    N = len(A)
    for i in range(N):
        k = np.argmax(np.abs(A[i:, i]))
        if k > 0:
            A[i + k], A[i] = np.copy(A[i]), np.copy(A[i + k])
        m = A[i, i:] / A[i, i]
        for j in range(i + 1, N):
            A[j, i:] -= A[j, i] * m
            A[j, :i] = 0.
    for i in range(N - 1, -1, -1):
        for j in range(N - 1, i, -1):
            A[i, -1] -= A[i, j] * A[j, -1]
        A[i, -1] /= A[i, i]

    print("A solução do terceiro sistema linear é:\n", A[:, -1])


print('Método Eliminação de Gauss')
print(gauss_3_3(l[2], Y[2]))
print()
print('-------------------------------------------------------')