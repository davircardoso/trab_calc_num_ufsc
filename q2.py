from math import exp
import numpy as np

e = exp(1)


def f(t):
    return 2 * e ** (-t / 2) / 3 ** (1 / 2) * np.sin(3 ** (1 / 2) * t / 2)


def df(t):
    return - ((e ** (t/2)) * (np.sin(np.sqrt(3) * t / 2)))/np.sqrt(3) + (e ** (-t/2)) * np.cos(np.sqrt(3) * t / 2)


def bisseccao(F, A, B, TOL ):
    """
      Encontra a raiz de F (em [A,B]) com um erro
      menor que TOL usando o algoritmo da Bissecção

      Recebe:
        F   => Função para encontrar a raiz
        A,B => Intervalo de busca da raiz
        TOL => Valor máximo do Erro (tolerância)

      Retorna:
        p   => Raiz encontrada
        E   => Erro
    """
    E = (B-A)/2.0
    p = (A+B)/2.0
    count = 0
    while E > TOL:
        count += 1
        if F(A)*F(p) < 0:
            B = p
        else:
            A = p
        E = (B-A)/2.0
        p = (A+B)/2.0

    return p, count


def fp(a, b):
    c = 1
    n = 0
    X = []
    Y = []
    while (f(c) > 10**-12) or (f(c) < -(10**-12)):
        c = b - ((f(b)*(b-a))/(f(b) - f(a)))
        n += 1
        X.append(n)
        Y.append(c)
        if c == a or c == b:
            return c, n, X, Y
        if f(c)*f(a) < 0:
            b = c
        elif f(c)*f(b) < 0:
            a = c

    return c, n


def newton(f, Df, x0, epsilon, max_iter):
    """
    Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    """
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            #print('Iterações Newton =',n)
            return xn, n
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None


print(newton(f, df, 2, 10**(-6), 1000))

print(fp(2, 6))

print(bisseccao(f, 2, 6, 10**(-6)))
