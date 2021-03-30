from math import exp
import numpy as np

e = exp(1)


def f(t):
    return 2 * e ** (-t / 2) / 3 ** (1 / 2) * np.sin(3 ** (1 / 2) * t / 2)


def df(t):
    return - ((e ** (t/2)) * (np.sin(np.sqrt(3) * t / 2)))/np.sqrt(3) + (e ** (-t/2)) * np.cos(np.sqrt(3) * t / 2)

def bisection(f, a, b, tol):
    error = (b-a)/2
    p = (a+b)/2
    count = 0
    while error > tol:
        count += 1
        if f(a) * f(p) < 0:
            b = p
        else:
            a = p
        error = (b-a)/2
        p = (a+b)/2
    return p, count

def false_position(f, a, b, eps):
    c = 1
    counter = 0
    while (f(c) > eps) or (f(c) < -eps):
        c = b - ((f(b)*(b-a))/(f(b) - f(a)))
        counter += 1
        if c == a or c == b:
            return c, counter
        if f(c)*f(a) < 0:
            b = c
        elif f(c)*f(b) < 0:
            a = c
    return c, counter


def newton(f, Df, x0, epsilon, max_iter):
    xn = x0
    for n in range(0, max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            return xn, n
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    print('Exceeded maximum iterations. No solution found.')
    return None

def secant(f, x0, x1, eps):
    f_x0 = f(x0)
    f_x1 = f(x1)
    counter = 0
    while abs(f_x1) > eps and counter < 100:
        denominator = float(f_x1 - f_x0) / (x1 - x0)
        x = x1 - float(f_x1) / denominator
        x0 = x1
        x1 = x
        f_x0 = f_x1
        f_x1 = f(x1)
        counter += 1
    if abs(f_x1) > eps:
        counter = -1
    return x, counter


print(secant(f, 2, 5, 10**(-6)))

print(newton(f, df, 2, 10**(-6), 1000))

print(false_position(f, 2, 6, 10**(-6)))

print(bisection(f, 2, 6, 10**(-6)))
