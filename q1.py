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
    t_residual_newt = []
    r_residual_newt = []
    tk = []
    xn = x0
    fxn = f(xn)
    count = 0
    tk.append(xn)
    while abs(fxn) > eps:
        tk.append(xn)
        t_residual_newt.append(xn)
        r_residual_newt.append(f(xn))
        count += 1
        fxn = f(xn)
        Dfxn = Df(xn)
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    return xn, count, tk, t_residual_newt, r_residual_newt

def secant(f, a, b, eps):
    f_a = f(a)
    f_b = f(b)
    t_res = []
    ta = []
    tb = []
    r_res = []
    count = 0
    while abs(f_b) > eps:
        ta.append(a)
        tb.append(b)
        df_aprox = a - (f(a) * (b - a) / (f(b) - f(a)))
        count += 1
        denominator = float(f_b - f_a) / (b - a)
        p = b - float(f_b) / denominator
        a = b
        b = p
        f_a = f_b
        f_b = f(b)
        r_res.append(f(df_aprox))
        t_res.append(b)

    return p, count, ta, tb, r_res, t_res

tol = 10**(-6) # tolerância máxima
bisec = bisection(f, 2, 6, tol)
fp = false_position(f, 2, 6, tol)
newt = newton(f, df, 2, tol)
sec = secant(f, 2, 5, tol)

# QUESTÃO 1.2
# print(bisec[0])
# print(fp[0])
# print(newt[0])
# print(sec[0])

#MOSTRANDO OS VALORES
print("Método da bisecção: " + str(bisec[0]) + "; Iterações: " + str(bisec[1]))
print("Método da falsa posição: " + str(fp[0]) + "; Iterações: " + str(fp[1]))
print("Método de newton: " + str(newt[0]) + "; Iterações: " + str(newt[1]))
print("Método das secantes: " + str(sec[0]) + "; Iterações: " + str(sec[1]))

# QUESTÃO 1.3
# print(bisec[1])
# print(fp[1])
# print(newt[1])
# print(sec[1])

# QUESTÃO 1.4.1

# bissecção

bisec_a = [x[0] for x in bisec[2]] # lista de valores de 'a' a cada iteração
bisec_b = [x[1] for x in bisec[2]] # lista de valores de 'b' a cada iteração
bisec_k = range(len(bisec_a)) # lista de iterações (0 a n)

# exibição do gráfico

bisec_graph_ka = plt.plot(bisec_k, bisec_a)
bisec_graph_kb = plt.plot(bisec_k, bisec_b)
bisec_graph_ka = plt.scatter(bisec_k, bisec_a)
bisec_graph_kb = plt.scatter(bisec_k, bisec_b)
plt.grid(True)
plt.legend((bisec_graph_ka, bisec_graph_kb), ('a', 'b'), loc='upper right')
plt.title('Método da bissecção')
plt.xlabel('k')
plt.show()

# falsa posição

fp_a = [x[0] for x in fp[2]] # lista de valores de 'a' a cada iteração
fp_b = [x[1] for x in fp[2]] # lista de valores de 'b' a cada iteração
fp_k = range(len(fp_a)) # lista de iterações (0 a n)

# exibição do gráfico

fp_graph_ka = plt.plot(fp_k, fp_a)
fp_graph_kb = plt.plot(fp_k, fp_b)
fp_graph_ka = plt.scatter(fp_k, fp_a)
fp_graph_kb = plt.scatter(fp_k, fp_b)
plt.grid(True)
plt.legend((fp_graph_ka, fp_graph_kb), ('a', 'b'), loc='upper right')
plt.title('Método da falsa posição')
plt.xlabel('k')
plt.show()

# método de newton

list_count_newt = list(range(newt[1]+1))

plt.xlabel('K iterações')
plt.ylabel('Tk')
plt.title('Tk Método de Newton')
plt.grid(True)
plt.gca().set_xlim([0, 25])
plt.gca().set_ylim([1.5, 4])
plt.plot(list_count_newt, newt[2], c='b')
plt.scatter(list_count_newt, newt[2], c='r')
plt.show()

# método da secante

list_count_sec = list(range(sec[1]))

plt.xlabel('K iterações')
plt.ylabel('Ta e Tb')
plt.title('Ta e Tb Método da Secante')
plt.grid(True)
plt.gca().set_xlim([0, 7])
plt.gca().set_ylim([-5, 7])
ta = plt.plot(list_count_sec, sec[2], c='b')
tb = plt.plot(list_count_sec, sec[3], c='orange')
ta = plt.scatter(list_count_sec, sec[2], c='b')
tb = plt.scatter(list_count_sec, sec[3], c='orange')
plt.legend((ta, tb), ('Ta', 'Tb'), loc='upper right')
plt.show()

# 1.5 - Newton Resíduo

k_n = list(range(newt[1]+1))

plt.xlabel('tempo')
plt.ylabel('R(t) e aproximação')
plt.title('Tk Método de Newton')
plt.grid(True)
plt.gca().set_xlim([1.5, 5])
plt.gca().set_ylim([-0.5, 0.6])
plt.plot(np.linspace(0,10), f(np.linspace(0,10)), c='lightblue')
plt.plot(newt[3], newt[4], c='r')
plt.scatter(newt[3], newt[4], c='b')
plt.show()

for i in range(len(newt[2])):
    newt[2][i] = newt[2][i] - max(newt[2])
plt.plot(k_n, newt[2], c='orange')
plt.scatter(k_n, newt[2], c='b')
plt.xlabel('iterações')
plt.ylabel('Aproximação do tempo de pico')
plt.title('Aproximação ao tempo de pico Método de Newton')
plt.grid(True)
plt.show()

# Secante Resíduo

plt.xlabel('tempo')
plt.ylabel('R(t) e aproximação')
plt.title('Tk Método da Secante')
plt.grid(True)
plt.gca().set_xlim([2, 5])
plt.gca().set_ylim([-1, 2])
plt.plot(np.linspace(2,5), f(np.linspace(2,5)), c='g')
plt.plot(sec[5], sec[4], c='r')
plt.scatter(sec[5], sec[4], c='b')
plt.show()

k = list(range(sec[1]))

val = sec[2][0]
for i in range(len(sec[2])):
    if sec[2][i] == val:
        sec[3][i] = sec[3][i] - max(sec[3])
    else:
        sec[2][i] = sec[2][i]-max(sec[2])
        val = sec[2][i]
plt.plot(k, sec[2], c='g')
plt.plot(k,sec[3], c='r')
plt.scatter(k, sec[2], c='g')
plt.scatter(k,sec[3] , c='r')
plt.xlabel('iterações')
plt.ylabel('Aproximação do tempo de pico')
plt.title('Aproximação ao tempo de pico Método da Secante')
plt.grid(True)
plt.show()


#Convergência 1.7

R_T_Newt, R_T_Sec = 0, 0
plt.plot(np.linspace(0,10),f(np.linspace(0,10)),'b')
R_T_Newt = f(1.6088)
plt.scatter(1.6088, R_T_Newt, color='r')
R_T_Sec = f(4.5194)
plt.scatter(4.5194, R_T_Sec, color='g')
plt.xlabel('tempo')
plt.ylabel('R(t)')
plt.title('Pontos extremos de convergência ao pico no R(t)')
plt.grid(True)
plt.show()


print(sec[5],sec[4])