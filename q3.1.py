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