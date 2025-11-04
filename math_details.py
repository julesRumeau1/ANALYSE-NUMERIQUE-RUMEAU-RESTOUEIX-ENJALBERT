import sympy as sp
import numpy as np


def pretty_polynomial(xi, yi, method='lagrange'):
    """Renvoie le polynôme (développé) en LaTeX pour Lagrange ou Newton, à partir de nœuds xi, valeurs yi."""
    x = sp.Symbol('x')
    xi = np.asarray(xi, float)
    yi = np.asarray(yi, float)
    n = len(xi)

    if method == 'lagrange':
        expr = 0
        for i in range(n):
            Li = 1
            for j in range(n):
                if j != i:
                    Li *= (x - xi[j]) / (xi[i] - xi[j])
            expr += yi[i] * Li
        return sp.latex(sp.expand(expr))

    if method == 'newton':
        coef = yi.copy()
        for j in range(1, n):
            coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (xi[j:n] - xi[0:n-j])
        expr = coef[0]
        prod = 1
        for i in range(1, n):
            prod *= (x - xi[i-1])
            expr += coef[i] * prod
        return sp.latex(sp.expand(expr))

    raise ValueError("method must be 'lagrange' or 'newton'.")


def _fmt_float(v):  # format compact pour LaTeX
    return f"{v:.6g}"


def latex_lagrange_basis(x_nodes, m=5, var="x"):
    """Retourne la liste des expressions LaTeX des bases de Lagrange ℓ_0..ℓ_{m-1} construites sur les m premiers nœuds."""
    x = np.asarray(x_nodes, float)
    m = min(m, len(x))
    out = []
    for i in range(m):
        num, den = [], []
        for j in range(m):
            if j == i: 
                continue
            num.append(fr"({var} - {_fmt_float(x[j])})")
            den.append(fr"({_fmt_float(x[i])} - {_fmt_float(x[j])})")
        out.append(r"\ell_{%d}(%s) = \frac{%s}{%s}" % (i, var, " \cdot ".join(num) or "1", " \cdot ".join(den) or "1"))
    return out


def latex_newton_details(x_nodes, y_nodes, m=5, var="x"):
    """
    Retourne (liste des a_k en LaTeX, polynôme de Newton factorisé en LaTeX)
    pour les m premiers nœuds.
    """
    x = np.asarray(x_nodes, float)
    y = np.asarray(y_nodes, float)
    m = min(m, len(x))

    coef = y.copy()
    a = [coef[0]]
    for j in range(1, m):
        coef[j:] = (coef[j:] - coef[j-1:-1]) / (x[j:] - x[:len(x)-j])
        a.append(coef[j])

    a_latex = [fr"a_{k} = {_fmt_float(a[k])}" for k in range(m)]

    terms = [_fmt_float(a[0])]
    for k in range(1, m):
        prod = " ".join([fr"({var} - {_fmt_float(x[j])})" for j in range(k)])
        terms.append(fr"{_fmt_float(a[k])} {prod}")
    poly = r"P_{%d}(%s) = " % (m-1, var) + " + ".join(terms)

    return a_latex, poly
