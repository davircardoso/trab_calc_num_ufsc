from math import exp
import numpy as np
import matplotlib.pyplot as plt

e = exp(1)

def f(t):
    return 2 * e ** (-t / 2) / 3 ** (1 / 2) * np.sin(3 ** (1 / 2) * t / 2)

def df(t):
    return - ((e ** (t/2)) * (np.sin(np.sqrt(3) * t / 2)))/np.sqrt(3) + (e ** (-t/2)) * np.cos(np.sqrt(3) * t / 2)

def bisection(f, a, b, tol):
    p = (a+b)/2
    count = 0
    x = []
    while abs(f(p)) > tol:
        x.append((a, b))
        count += 1
        if p in (a, b):
            return p, count
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        p = (a + b) / 2
    return p, count, x

def false_position(f, a, b, tol):
    p = 1
    count = 0
    x = []
    while abs(f(p)) > tol:
        p = a + ((f(a) * (b - a)) / (f(a) - f(b)))
        x.append((a, b))
        count += 1
        if p in (a, b):
            return p, count
        if f(p) * f(a) < 0:
            b = p
        elif f(p) * f(b) < 0:
            a = p
    return p, count, x

def newton(f, Df, x0, eps):
    xn = x0
    fxn = f(xn)
    count = 0
    while abs(fxn) > eps:
        count += 1
        fxn = f(xn)
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    return xn, count

def secant(f, a, b, eps):
    f_a = f(a)
    f_b = f(b)
    count = 0
    while abs(f_b) > eps:
        count += 1
        denominator = float(f_b - f_a) / (b - a)
        p = b - float(f_b) / denominator
        a = b
        b = p
        f_a = f_b
        f_b = f(b)
    return p, count

tol = 10**(-6) # tolerância máxima
bisec = bisection(f, 2, 6, tol)
fp = false_position(f, 2, 6, tol)
newt = newton(f, df, 2, tol)
sec = secant(f, 2, 5, tol)

# QUESTÃO 1.2
print(bisec[0])
print(fp[0])
print(newt[0])
print(sec[0])

# QUESTÃO 1.3
print(bisec[1])
print(fp[1])
print(newt[1])
print(sec[1])

# QUESTÃO 1.4.1
bisec_ab = bisec[2]
bisec_k = range(len(bisec_ab))
bisec_a = [x[0] for x in bisec_ab]
bisec_b = [x[1] for x in bisec_ab]
bisec_graph_ka = plt.scatter(bisec_k, bisec_a)
bisec_graph_kb = plt.scatter(bisec_k, bisec_b)
plt.grid(True)
plt.legend((bisec_graph_ka, bisec_graph_kb), ('a', 'b'), loc='upper right')
plt.title('Método da bissecção')
plt.xlabel('k')
plt.show()

fp_ab = fp[2]
fp_k = range(len(fp_ab))
fp_a = [x[0] for x in fp_ab]
fp_b = [x[1] for x in fp_ab]
fp_graph_ka = plt.scatter(fp_k, fp_a)
fp_graph_kb = plt.scatter(fp_k, fp_b)
plt.grid(True)
plt.legend((fp_graph_ka, fp_graph_kb), ('a', 'b'), loc='upper right')
plt.title('Método da falsa posição')
plt.xlabel('k')
plt.show()