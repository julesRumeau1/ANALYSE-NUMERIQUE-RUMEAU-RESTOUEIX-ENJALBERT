import sympy as sp
import numpy as np

# ---------------------------------------------------------------
# 1️⃣ LAGRANGE CLASSIQUE
# ---------------------------------------------------------------
def lagrange_classique(xi, yi, x_eval):
    xi, yi = np.array(xi, dtype=float), np.array(yi, dtype=float)
    n = len(xi)
    P = np.zeros_like(x_eval, dtype=float)
    for i in range(n):
        L = np.ones_like(x_eval)
        for j in range(n):
            if j != i:
                L *= (x_eval - xi[j]) / (xi[i] - xi[j])
        P += yi[i] * L
    return P


# ---------------------------------------------------------------
# 2️⃣ NEWTON (différences divisées)
# ---------------------------------------------------------------
def newton_dif_div(xi, yi, x_eval):
    xi, yi = np.array(xi, dtype=float), np.array(yi, dtype=float)
    n = len(xi)
    coef = yi.copy()
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xi[j:n] - xi[0:n-j])
    # évaluation
    P = np.zeros_like(x_eval)
    for c, xj in zip(coef[::-1], xi[::-1]):
        P = P * (x_eval - xj) + c
    return P


# ---------------------------------------------------------------
# 3️⃣ FORMULES SYMBOLIQUES POUR AFFICHAGE
# ---------------------------------------------------------------
def pretty_polynomial(xi, yi, method='lagrange'):
    x = sp.Symbol('x')
    n = len(xi)
    expr = 0

    if method == 'lagrange':
        for i in range(n):
            Li = 1
            for j in range(n):
                if j != i:
                    Li *= (x - xi[j]) / (xi[i] - xi[j])
            expr += yi[i] * Li
        return sp.latex(sp.expand(expr))

    elif method == 'newton':
        coef = yi.copy()
        for j in range(1, n):
            coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xi[j:n] - xi[0:n-j])
        expr = coef[0]
        prod = 1
        for i in range(1, n):
            prod *= (x - xi[i - 1])
            expr += coef[i] * prod
        return sp.latex(sp.expand(expr))
