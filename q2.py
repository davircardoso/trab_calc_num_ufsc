import math
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
from autograd import jacobian

# Valor de x
x1,x2,x3,x4,x5 = sp.symbols("x1 x2 x3 x4 x5")
x = sp.Matrix([[x1],[x2],[x3],[x4],[x5],[1],[1],[1]])
x0 = sp.Matrix([[0.2],[0.5],[0.85],[0.93],[0.22]])

# Matrizes A, g, A*x, F, JacF
A = sp.Matrix([[-3.933, 0.107, 0.126, 0, -9.99, 0, -45.83, -7.64],[0, -0.987, 0, -22.95, 0, -28.37, 0, 0],[0.002, 0, -0.235, 0, 5.67, 0, -0.921, -6.51],[0, 1, 0, -1, 0, -0.168, 0, 0],[0, 0, -1, 0, -0.196, 0, -0.0071, 0]])
g = sp.Matrix([[-0.727*x2*x3+8.39*x3*x4-684.4*x4*x5+63.5*x4*x2],[0.949*x1*x3+0.173*x1*x5],[-0.716*x1*x2-1.578*x1*x4 + 1.132*x4*x2],[-x1*x5],[x1*x4]])
Ax = sp.Matrix(np.matmul(A,x))
F = sp.Matrix(Ax + g)
JacF = sp.Matrix(F.jacobian(x[0:5]))

# Método de Newton Raphson Multivariável
def Newton(f,z,it):
    xv = sp.Matrix(z)
    for i in range (it):
        xn = xv - np.matmul(JacF.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]).inv(),F.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]))
        if ((F.subs([(x1,xn[0,0]),(x2,xn[1,0]),(x3,xn[2,0]),(x4,xn[3,0]),(x5,xn[4,0])])).norm() < 1e-6):
            print('convergiu em ', i+1, ' iterações, com x =', xn)
            break
        xv = xn

def Tabela(f,z,it):
    coluna = 'x1 x2 x3 x4 x5 residuo'.split()
    linha = '1 2 3 4 5 6 7 8 9 10'.split()
    dados = []
    xv = sp.Matrix(z)
    for i in range (it):
        xn = xv - np.matmul(JacF.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]).inv(),F.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]))
        dados.append([xn[0,0],xn[1,0],xn[2,0],xn[3,0],xn[4,0],(F.subs([(x1,xn[0,0]),(x2,xn[1,0]),(x3,xn[2,0]),(x4,xn[3,0]),(x5,xn[4,0])])).norm()])
        if (i == (it-1)):
            dados = np.array(dados)
            dados = dados.reshape(len(linha),len(coluna))
            tabela = pd.DataFrame(data=dados,index=linha,columns=coluna)
            print(tabela)
            break
        xv = xn

def Grafico(f,z,it):
    xv = sp.Matrix(z)
    x=[]
    res=[]
    erro=[]
    iteracoes=[]
    for i in range (it):
        xn = xv - np.matmul(JacF.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]).inv(),F.subs([(x1,xv[0,0]),(x2,xv[1,0]),(x3,xv[2,0]),(x4,xv[3,0]),(x5,xv[4,0])]))
        x.append([xn[0,0],xn[1,0],xn[2,0],xn[3,0],xn[4,0]])
        res.append([(F.subs([(x1,xn[0,0]),(x2,xn[1,0]),(x3,xn[2,0]),(x4,xn[3,0]),(x5,xn[4,0])])).norm()])
        erro.append([abs(3.00979089670122-xn[0,0]),abs(-28.1949897945902-xn[1,0]),abs(1.79853794936703-xn[2,0]),abs(-0.0134581819116551-xn[3,0]),abs(-9.41910338148413-xn[4,0])])
        iteracoes.append(i)
        if (i == (it-1)):
            fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
            ax1.plot(iteracoes,x)
            ax2.plot(iteracoes,res)
            ax3.plot(iteracoes,erro)
            plt.show()
            break
        xv=xn

Newton(F,x0,100)
Tabela(F,x0,10)
Grafico(F,x0,13)
