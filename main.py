#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
main.py
Projet : Reconstruction de trajectoires GPS par interpolation (tout dans un seul fichier)

Usage :
    python main.py

Ce script :
- génère une trajectoire 2D "réelle" paramétrée par t (fonction synthétique),
- échantillonne N points (option : bruit gaussien),
- construit 3 types d'interpolateurs séparés (x(t) et y(t)) :
    * Lagrange (forme barycentrique)
    * Newton (différences divisées)
    * Spline cubique (scipy.interpolate.CubicSpline)
- évalue les reconstructions sur une grille dense,
- calcule MAE et RMSE (erreur euclidienne point à point),
- trace la trajectoire réelle vs reconstruites et l'évolution des erreurs en fonction du nombre de points,
- affiche et sauvegarde un tableau des erreurs (pandas DataFrame).

Commentaires : je commente le code au niveau "étudiant informatique" : suffisant pour comprendre sans être verbeux.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# tentative d'import de SciPy (nécessaire pour la spline). On abort proprement si absent.
try:
    from scipy.interpolate import CubicSpline
except Exception as e:
    print("Erreur : scipy n'est pas installé ou introuvable.")
    print("Installe scipy avec : pip install scipy")
    raise

# -------------------------
# Fonctions pour générer données (traj réelle puis échantillonnage)
# -------------------------

def generate_ground_truth(num_points=1000, t_min=0.0, t_max=10.0):
    """
    Génère une trajectoire 2D lisse (x(t), y(t)) échantillonnée sur une grille dense.
    - num_points : nombre de points de la grille d'évaluation (dense)
    - [t_min, t_max] : intervalle de paramètre t
    
    Retour :
        t_dense (np.ndarray), x_dense (np.ndarray), y_dense (np.ndarray)
    La trajectoire choisie est volontairement non triviale (somme de sin/cos) pour être représentative.
    """
    t = np.linspace(t_min, t_max, num_points)
    # paramétrisation synthétique : mélange d'ondes (très classique pour tests)
    x = 2.0 * np.cos(0.7 * t) + 0.5 * np.sin(2.0 * t) + 0.2 * np.sin(5.0 * t)
    y = 1.5 * np.sin(0.9 * t) + 0.4 * np.cos(3.0 * t) + 0.15 * np.cos(6.0 * t)
    return t, x, y

def sample_points_from_dense(t_dense, x_dense, y_dense, n_samples=100, noise_std=0.0, random_seed=None):
    """
    Echantillonne n_samples points uniformément (selon t) à partir de la trajectoire dense.
    Option : ajoute du bruit gaussien isotrope aux coordonnées (sigma = noise_std).
    Retour : ti, xi, yi (tableau de taille n_samples)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    # choisir indices uniformes (évite clusters)
    idx = np.linspace(0, len(t_dense) - 1, n_samples).astype(int)
    ti = t_dense[idx]
    xi = x_dense[idx].copy()
    yi = y_dense[idx].copy()
    if noise_std > 0:
        xi += np.random.normal(scale=noise_std, size=xi.shape)
        yi += np.random.normal(scale=noise_std, size=yi.shape)
    return ti, xi, yi

# -------------------------
# Interpolations : Lagrange (barycentric), Newton, Spline cubique
# -------------------------


def lagrange_interpolator(xi, yi):
    """
    Interpolation polynomiale de Lagrange (version simple, directe).
    Retourne une fonction f(t) qui évalue le polynôme interpolant
    passant par tous les points (xi, yi).

    Attention : complexité O(n^2) à l’évaluation, donc lent si n est grand.
    Suffisant pour des TP (jusqu’à 200 points environ).
    """
    xi = np.asarray(xi)
    yi = np.asarray(yi)

    def f(t):
        t = np.asarray(t, dtype=float)
        n = len(xi)
        res = np.zeros_like(t, dtype=float)

        # double boucle : somme sur i des y_i * L_i(t)
        for i in range(n):
            Li = np.ones_like(t)
            for j in range(n):
                if j != i:
                    Li *= (t - xi[j]) / (xi[i] - xi[j])
            res += yi[i] * Li
        return res

    return f

def newton_interpolator(x_points, y_points):
    """
    Interpolation de Newton avec différences divisées récursives.
    x_points : liste ou array des x_i
    y_points : liste ou array des y_i

    Retourne une fonction f(t) qui évalue le polynôme de Newton passant par les points.
    """

    x_points = np.asarray(x_points, dtype=float)
    y_points = np.asarray(y_points, dtype=float)
    n = len(x_points)

    # fonction récursive pour f[x_i, ..., x_j]
    def divided_diff(x_sub, y_sub):
        if len(x_sub) == 1:
            return y_sub[0]
        return (divided_diff(x_sub[1:], y_sub[1:]) - divided_diff(x_sub[:-1], y_sub[:-1])) / (x_sub[-1] - x_sub[0])

    # calcul des coefficients a_i = f[x_0, ..., x_i]
    coeffs = np.array([divided_diff(x_points[:i+1], y_points[:i+1]) for i in range(n)], dtype=float)

    # fonction d'évaluation du polynôme de Newton
    def f(t):
        t = np.asarray(t, dtype=float)
        res = np.zeros_like(t, dtype=float)
        for i in range(n-1, -1, -1):
            res = res * (t - x_points[i]) + coeffs[i]
        return res

    return f


def spline_cubic_interpolator(xi, yi, bc_type='natural'):
    """
    Wrapper sur scipy.interpolate.CubicSpline pour renvoyer une fonction callable.
    bc_type : conditions aux bords, 'natural' par défaut.
    """
    cs = CubicSpline(xi, yi, bc_type=bc_type)
    return lambda t: cs(t)

# -------------------------
# Mesures d'erreur (MAE, RMSE) sur trajectoire 2D
# -------------------------

def compute_pointwise_error(x_true, y_true, x_pred, y_pred):
    """
    Calcule l'erreur euclidienne point à point entre (x_true, y_true) et (x_pred, y_pred).
    Renvoie tableau d'erreurs (shape = n_points).
    """
    err = np.sqrt((x_true - x_pred)**2 + (y_true - y_pred)**2)
    return err

def MAE(err_array):
    return np.mean(np.abs(err_array))

def RMSE(err_array):
    return np.sqrt(np.mean(err_array**2))

# -------------------------
# Routine principale d'évaluation comparative
# -------------------------

def evaluate_reconstruction(t_dense, x_dense, y_dense, ti, xi, yi, eval_t=None, methods=('lagrange', 'newton', 'spline')):
    """
    Pour un ensemble d'échantillons (ti,xi,yi), construit les interpolateurs demandés,
    les évalue sur eval_t (ou t_dense si None), et renvoie les résultats et métriques.
    
    Retour :
        results : dict method -> dict {
            'x_pred': ..., 'y_pred': ..., 'err_pointwise': ..., 'mae': float, 'rmse': float
        }
    """
    if eval_t is None:
        eval_t = t_dense

    results = {}

    # construire interpolateurs séparés pour chaque méthode
    for method in methods:
        if method == 'lagrange':
            fx = lagrange_interpolator(ti, xi)
            fy = lagrange_interpolator(ti, yi)
        elif method == 'newton':
            fx = newton_interpolator(ti, xi)
            fy = newton_interpolator(ti, yi)
        elif method == 'spline':
            fx = spline_cubic_interpolator(ti, xi)
            fy = spline_cubic_interpolator(ti, yi)
        else:
            raise ValueError(f"Méthode inconnue : {method}")

        # évaluer
        x_pred = fx(eval_t)
        y_pred = fy(eval_t)

        # erreurs (comparées à la trajectoire dense de référence)
        err = compute_pointwise_error(x_dense_interp(eval_t, t_dense, x_dense),
                                      y_dense_interp(eval_t, t_dense, y_dense),
                                      x_pred, y_pred)
        mae = MAE(err)
        rmse = RMSE(err)

        results[method] = {
            'x_pred': x_pred,
            'y_pred': y_pred,
            'err_pointwise': err,
            'mae': mae,
            'rmse': rmse
        }
    return results

# -------------------------
# Helpers : interpolation simple pour obtenir la "vraie" valeur (sur t_dense) aux t demandés
# (on utilise une spline pour la référence - la trajectoire dense est déjà sur une grille,
#  mais pour sécurité on retourne une interpolation continue)
# -------------------------

def x_dense_interp(t_query, t_dense, x_dense):
    """Interpole linéairelement (rapide) la valeur x vraie pour t_query"""
    return np.interp(t_query, t_dense, x_dense)

def y_dense_interp(t_query, t_dense, y_dense):
    """Interpole linéaireement (rapide) la valeur y vraie pour t_query"""
    return np.interp(t_query, t_dense, y_dense)

# -------------------------
# Fonction pour étudier l'évolution de l'erreur en fonction du nombre de points
# -------------------------

def error_vs_npoints(t_dense, x_dense, y_dense, n_list, noise_std=0.0, random_seed=None, methods=('lagrange', 'newton', 'spline')):
    """
    Pour chaque nombre de points dans n_list :
        - échantillonne ces points (avec même random_seed si fourni),
        - construit interpolateurs,
        - calcule MAE & RMSE sur la grille t_dense.
    Retourne un DataFrame avec colonnes :
        ['n_points', 'method', 'mae', 'rmse']
    """
    rows = []
    for n in n_list:
        ti, xi, yi = sample_points_from_dense(t_dense, x_dense, y_dense, n_samples=n, noise_std=noise_std, random_seed=random_seed)
        # évaluer chaque méthode
        for method in methods:
            if method == 'lagrange':
                fx = lagrange_interpolator(ti, xi)
                fy = lagrange_interpolator(ti, yi)
            elif method == 'newton':
                fx = newton_interpolator(ti, xi)
                fy = newton_interpolator(ti, yi)
            elif method == 'spline':
                fx = spline_cubic_interpolator(ti, xi)
                fy = spline_cubic_interpolator(ti, yi)
            else:
                continue

            x_pred = fx(t_dense)
            y_pred = fy(t_dense)
            err = compute_pointwise_error(x_dense, y_dense, x_pred, y_pred)
            rows.append({'n_points': n, 'method': method, 'mae': MAE(err), 'rmse': RMSE(err)})
    df = pd.DataFrame(rows)
    return df

# -------------------------
# Visualisations (plots)
# -------------------------

def plot_trajectories(t_dense, x_dense, y_dense, ti, xi, yi, results, output_dir='outputs', show=True):
    """
    Trace la trajectoire réelle et les trajectoires reconstruites (pour chaque méthode).
    Sauvegarde une figure.
    """
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.plot(x_dense, y_dense, label='Trajectoire réelle (dense)', linewidth=2)
    plt.scatter(xi, yi, color='k', s=15, label=f'Points échantillons (n={len(ti)})', zorder=10)
    colors = {'lagrange':'C0', 'newton':'C1', 'spline':'C2'}
    for method, res in results.items():
        plt.plot(res['x_pred'], res['y_pred'], linestyle='--', label=f'{method} (MAE={res["mae"]:.3e}, RMSE={res["rmse"]:.3e})', color=colors.get(method, None))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectoire réelle vs reconstructions')
    plt.legend()
    plt.axis('equal')
    outpath = os.path.join(output_dir, 'trajectoires_comparaison.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"[INFO] Figure sauvegardée : {outpath}")

def plot_error_evolution(df_errors, output_dir='outputs', show=True):
    """
    Trace évolution de MAE & RMSE en fonction du nombre de points (groupé par méthode).
    df_errors attend un DataFrame (colonnes : n_points, method, mae, rmse)
    """
    os.makedirs(output_dir, exist_ok=True)
    # MAE
    plt.figure(figsize=(8,5))
    for method, g in df_errors.groupby('method'):
        g_sorted = g.sort_values('n_points')
        plt.plot(g_sorted['n_points'], g_sorted['mae'], marker='o', label=method)
    plt.xlabel('Nombre de points utilisés')
    plt.ylabel('MAE')
    plt.title('Évolution de la MAE en fonction du nombre de points')
    plt.legend()
    plt.grid(True)
    outpath = os.path.join(output_dir, 'mae_vs_npoints.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"[INFO] Figure sauvegardée : {outpath}")

    # RMSE
    plt.figure(figsize=(8,5))
    for method, g in df_errors.groupby('method'):
        g_sorted = g.sort_values('n_points')
        plt.plot(g_sorted['n_points'], g_sorted['rmse'], marker='o', label=method)
    plt.xlabel('Nombre de points utilisés')
    plt.ylabel('RMSE')
    plt.title('Évolution du RMSE en fonction du nombre de points')
    plt.legend()
    plt.grid(True)
    outpath = os.path.join(output_dir, 'rmse_vs_npoints.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    if show:
        plt.show()
    plt.close()
    print(f"[INFO] Figure sauvegardée : {outpath}")

# -------------------------
# Main : exécution par défaut / démonstration
# -------------------------

def main():
    # --- paramètres que tu peux modifier rapidement ---
    dense_points = 1000         # grille dense pour la "vérité"
    sample_N = 20              # nombre de points échantillonnés (entre 50 et 200 demandé) !!! mal opti donc 20 pour le moment
    noise_std = 0.02            # bruit gaussien sur xi, yi (0.0 = pas de bruit)
    random_seed = 42            # reproductibilité
    methods = ('lagrange', 'newton', 'spline')
    output_dir = 'outputs'
    # --------------------------------------------------

    print("Génération de la trajectoire de référence...")
    t_dense, x_dense, y_dense = generate_ground_truth(num_points=dense_points, t_min=0.0, t_max=12.0)

    print(f"Echantillonnage de {sample_N} points (bruit std = {noise_std})...")
    ti, xi, yi = sample_points_from_dense(t_dense, x_dense, y_dense, n_samples=sample_N, noise_std=noise_std, random_seed=random_seed)

    # évaluer chaque méthode sur la grille dense
    print("Construction et évaluation des interpolateurs...")
    results = {}
    for method in methods:
        if method == 'lagrange':
            fx = lagrange_interpolator(ti, xi)
            fy = lagrange_interpolator(ti, yi)
        elif method == 'newton':
            fx = newton_interpolator(ti, xi)
            fy = newton_interpolator(ti, yi)
        elif method == 'spline':
            fx = spline_cubic_interpolator(ti, xi)
            fy = spline_cubic_interpolator(ti, yi)
        else:
            continue

        x_pred = fx(t_dense)
        y_pred = fy(t_dense)
        err = compute_pointwise_error(x_dense, y_dense, x_pred, y_pred)
        results[method] = {
            'x_pred': x_pred,
            'y_pred': y_pred,
            'err_pointwise': err,
            'mae': MAE(err),
            'rmse': RMSE(err)
        }
        print(f"  -> {method} : MAE = {results[method]['mae']:.6f}, RMSE = {results[method]['rmse']:.6f}")

    # tracé comparatif
    plot_trajectories(t_dense, x_dense, y_dense, ti, xi, yi, results, output_dir=output_dir, show=True)

    # étude erreur vs nombre de points
    print("Calcul de l'évolution des erreurs en fonction du nombre de points...")
    # choisir quelques tailles de sous-échantillonnage entre 10 et sample_N (ou upto 200)
    n_list = np.unique(np.round(np.linspace(10, min(200, sample_N), 10)).astype(int))
    df_err = error_vs_npoints(t_dense, x_dense, y_dense, n_list, noise_std=noise_std, random_seed=random_seed, methods=methods)

    # afficher & sauvegarder le tableau
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'erreurs_par_npoints.csv')
    df_err.to_csv(csv_path, index=False)
    print(f"[INFO] Table des erreurs sauvegardée : {csv_path}")
    print("\nTable (extrait) :")
    print(df_err.head(20))

    # tracer évolution
    plot_error_evolution(df_err, output_dir=output_dir, show=True)

    # impression résumé clair (direct)
    print("\nRésumé direct :")
    for method in methods:
        # prendre la ligne où n_points == sample_N
        row = df_err[(df_err['n_points'] == sample_N) & (df_err['method'] == method)]
        if not row.empty:
            mae = float(row['mae'].values[0])
            rmse = float(row['rmse'].values[0])
            print(f" - {method} (avec n={sample_N}) : MAE={mae:.6f}, RMSE={rmse:.6f}")
        else:
            # utiliser la valeur calculée plus haut sur la grille complète
            mae = results[method]['mae']
            rmse = results[method]['rmse']
            print(f" - {method} (évalué) : MAE={mae:.6f}, RMSE={rmse:.6f}")

    print("\nTerminé. Résultats et figures dans le dossier 'outputs'.")

if __name__ == '__main__':
    main()
