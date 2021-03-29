from math import exp
import numpy as np

e = exp(1)


def funcao(t):
    return 2 * e ** (-t / 2) / 3 ** (1 / 2) * np.sin(3 ** (1 / 2) * t / 2)


def bisseccao(f, a, b, prec):
    """ Método da bisseção para uma função f no intervalo [a,b]. """
    m = (a + b) / 2
    # Se já há precisão suficiente, retornamos o ponto médio do intervalo
    if abs(b - a) < prec: return m
    # Se f(m) == 0, achamos uma raiz exata!
    if f(m) == 0: return m

    # Senão, vamos por recorrência
    if f(m) * f(a) < 0:
        return bissecao(f, a, m, prec)
    else:
        return bissecao(f, m, b, prec)


print(bisseccao(funcao, 2, 6, 1e-6))