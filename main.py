
import numpy as np
import matplotlib.pyplot as plt

# tentative d'import de SciPy (nécessaire pour la spline). On abort proprement si absent.
try:
    from scipy.interpolate import CubicSpline
except Exception as e:
    print("Erreur : scipy n'est pas installé ou introuvable.")
    print("Installe scipy avec : pip install scipy")
    raise


# Génère une trajectoire 2D synthétique (x(t), y(t)), échantillonnée uniformément entre 0 et 10
def generate_ground_truth(n=1000):
    t = np.linspace(0, 10, n)
    x = np.cos(t) + 0.5 * np.sin(2 * t)
    y = np.sin(1.2 * t) + 0.3 * np.cos(3 * t)
    return t, x, y


# Interpolateur de Lagrange : retourne le polynôme passant par les points (xi, yi)
def lagrange(xi, yi):
    xi, yi = np.asarray(xi), np.asarray(yi)
    n = len(xi)

    def f(t):
        t = np.asarray(t)
        p = np.zeros_like(t, dtype=float)
        for i in range(n):
            L = np.ones_like(t)
            for j in range(n):
                if j != i:
                    L *= (t - xi[j]) / (xi[i] - xi[j])
            p += yi[i] * L
        return p

    return f


# Interpolateur de Newton : retourne le polynôme passant par les points (x_points, y_points)
def newton(x_points, y_points):
    x, y = np.asarray(x_points, float), np.asarray(y_points, float)
    n = len(x)

    # calcul des coefficients avec différences divisées
    coef = y.copy()
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j - 1:-1]) / (x[j:] - x[:n - j])

    # évaluation efficace par schéma de Horner généralisé
    def f(t):
        t = np.asarray(t, float)
        p = np.zeros_like(t)
        for c, xi in zip(coef[::-1], x[::-1]):
            p = p * (t - xi) + c
        return p

    return f




def spline_cubic_interpolator(xi, yi, bc_type='natural'):
    """
    Wrapper sur scipy.interpolate.CubicSpline pour renvoyer une fonction callable.
    bc_type : conditions aux bords, 'natural' par défaut.
    """
    cs = CubicSpline(xi, yi, bc_type=bc_type)
    return lambda t: cs(t)




# Trace la trajectoire réelle et les interpolations avec styles améliorés pour lisibilité
def plot_trajectories(x_dense, y_dense, lag, newt, spline, x_sample, y_sample):
    """
    x_dense, y_dense : trajectoire réelle
    lag, newt, spline : tuples (x_interp, y_interp) retournés par compute_interpolations
    x_sample, y_sample : points d'échantillon
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(9,6))

    # Trajectoire réelle
    plt.plot(x_dense, y_dense, 'k-', linewidth=2, alpha=0.5, label='Trajectoire réelle')

    # Interpolations
    plt.plot(lag[0], lag[1], 'r--', linewidth=1.5, label='Lagrange')
    plt.plot(newt[0], newt[1], 'g-.', linewidth=1.5, label='Newton')
    plt.plot(spline[0], spline[1], 'b-', linewidth=1, alpha=0.7, label='Spline cubique')

    # Points échantillons
    plt.scatter(x_sample, y_sample, c='orange', s=50, edgecolor='k', zorder=5, label='Points échantillons')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectoire réelle vs interpolations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Affiche les erreurs MAE et RMSE pour chaque interpolation
def plot_errors(errors, errors_rmse):
    """
    errors : dict avec MAE pour chaque méthode
    errors_rmse : dict avec RMSE pour chaque méthode
    """

    methods = list(errors.keys())
    mae_values = [errors[m] for m in methods]
    rmse_values = [errors_rmse[m] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(7,5))
    plt.bar(x - width/2, mae_values, width, label='MAE', color='skyblue')
    plt.bar(x + width/2, rmse_values, width, label='RMSE', color='salmon')
    plt.xticks(x, methods)
    plt.ylabel('Erreur')
    plt.title('Comparaison des erreurs')
    plt.legend()
    plt.show()



def compute_interpolations(t_sample, x_sample, y_sample, t_dense):
    from scipy.interpolate import CubicSpline

    # interpolations
    lag_x = lagrange(t_sample, x_sample)(t_dense)
    lag_y = lagrange(t_sample, y_sample)(t_dense)

    newt_x = newton(t_sample, x_sample)(t_dense)
    newt_y = newton(t_sample, y_sample)(t_dense)

    spline_x = CubicSpline(t_sample, x_sample)(t_dense)
    spline_y = CubicSpline(t_sample, y_sample)(t_dense)

    return (lag_x, lag_y), (newt_x, newt_y), (spline_x, spline_y)




# Calcule les erreurs MAE et RMSE entre les interpolations et la trajectoire réelle
def compute_errors(x_dense, y_dense, lag, newt, spline):
    import numpy as np

    def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
    def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred)**2))

    errors = {
        "Lagrange": (mae(x_dense, lag[0])+mae(y_dense, lag[1]))/2,
        "Newton": (mae(x_dense, newt[0])+mae(y_dense, newt[1]))/2,
        "Spline": (mae(x_dense, spline[0])+mae(y_dense, spline[1]))/2,
    }

    errors_rmse = {
        "Lagrange": (rmse(x_dense, lag[0])+rmse(y_dense, lag[1]))/2,
        "Newton": (rmse(x_dense, newt[0])+rmse(y_dense, newt[1]))/2,
        "Spline": (rmse(x_dense, spline[0])+rmse(y_dense, spline[1]))/2,
    }

    return errors, errors_rmse


if __name__ == "__main__":
    import numpy as np

    # --- Générer la trajectoire dense (ground truth) ---
    t_dense, x_dense, y_dense = generate_ground_truth(n=1000)
    
    # --- Échantillonner quelques points pour l'interpolation ---
    n_sample = int(input("Nombre de points (entre 50 et 200): "))  # nombre de points échantillon
    indices = np.linspace(0, len(t_dense)-1, n_sample, dtype=int)
    t_sample = t_dense[indices]
    x_sample = x_dense[indices]
    y_sample = y_dense[indices]

    # --- Calculer les interpolations ---
    lag, newt, spline = compute_interpolations(t_sample, x_sample, y_sample, t_dense)

    # --- Tracer les trajectoires ---
    plot_trajectories(x_dense, y_dense, lag, newt, spline, x_sample, y_sample)

    # --- Calculer les erreurs ---
    errors, errors_rmse = compute_errors(x_dense, y_dense, lag, newt, spline)

    # --- Afficher les erreurs ---
    print("MAE :", errors)
    print("RMSE :", errors_rmse)
    plot_errors(errors, errors_rmse)
