import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw
from scipy.interpolate import CubicSpline
from pyproj import Transformer
from math_details import pretty_polynomial  # gardé pour l'affichage symbolique

from scipy.interpolate import PchipInterpolator  # au début du fichier

def resample_equidistant(s, x, y, n):
    """
    s : longueur d’arc cumulative (croissante), shape (N,)
    x,y : coordonnées en mètres sur s
    n : nombre de points désirés (50–200)
    -> renvoie (s_new, x_new, y_new) échantillonnés à distances égales.
    """
    s0, s1 = float(s[0]), float(s[-1])
    s_new = np.linspace(s0, s1, n)
    # PCHIP évite les overshoots; natural CubicSpline marcherait aussi
    fx = PchipInterpolator(s, x)
    fy = PchipInterpolator(s, y)
    return s_new, fx(s_new), fy(s_new)


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Interpolation polynomiale 2D", layout="wide")
st.title("Interpolation polynomiale 2D — Lagrange • Newton • Spline")

# =========================
# Fonctions de ton script de base
# =========================
def generate_ground_truth(n=1000):
    t = np.linspace(0, 10, n)
    x = np.cos(t) + 0.5 * np.sin(2 * t)
    y = np.sin(1.2 * t) + 0.3 * np.cos(3 * t)
    return t, x, y

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

def newton(x_points, y_points):
    x, y = np.asarray(x_points, float), np.asarray(y_points, float)
    n = len(x)
    coef = y.copy()
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j - 1:-1]) / (x[j:] - x[:n - j])
    def f(t):
        t = np.asarray(t, float)
        p = np.zeros_like(t)
        for c, xi in zip(coef[::-1], x[::-1]):
            p = p * (t - xi) + c
        return p
    return f

def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred)**2)))

# =========================
# Outils trajets réels
# =========================
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

def get_route_osrm(start, end):
    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        f"{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if "routes" not in data or not data["routes"]:
        return []
    coords = data["routes"][0]["geometry"]["coordinates"]
    return [(lat, lon) for lon, lat in coords]

def latlon_to_xy(latlon):
    lons = [lon for lat, lon in latlon]
    lats = [lat for lat, lon in latlon]
    x, y = _transformer.transform(lons, lats)  # (lon,lat) -> (x,y) en m
    return np.asarray(x, float), np.asarray(y, float)

def arclength_param(x, y):
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    return s

def extract_two_markers(map_state):
    pts = []
    if not map_state:
        return pts
    drawings = map_state.get("all_drawings") or []
    if isinstance(drawings, dict):
        drawings = [drawings]
    for g in drawings:
        try:
            if g.get("type") == "marker":
                lon, lat = g["geometry"]["coordinates"]
                pts.append((lat, lon))
        except Exception:
            pass
    return pts[-2:]

# =========================
# Sidebar
# =========================
mode = st.sidebar.radio("Mode", ["Courbe synthétique", "Trajet réel (carte/OSRM)"])
n_sample = st.sidebar.slider("Nombre de points (échantillonnage des données)", 5, 200, 100, step=5)
stabilize = st.sidebar.checkbox("Stabiliser Lagrange/Newton (normaliser t)", value=True)

# helper normalisation pour stabiliser les polynômes
def normalize_t(nodes, t_eval):
    a = float(nodes[0]); b = float(nodes[-1])
    if b == a:
        th_nodes = np.zeros_like(nodes, float); th_eval = np.zeros_like(t_eval, float)
    else:
        th_nodes = (nodes - a) / (b - a)
        th_eval  = (t_eval - a) / (b - a)
    return th_nodes, th_eval

# =========================
# MODE 1 : Courbe synthétique (fidèle à ton script)
# =========================
if mode == "Courbe synthétique":
    t_dense, x_dense, y_dense = generate_ground_truth(n=1000)
    idx = np.linspace(0, len(t_dense) - 1, n_sample, dtype=int)
    t_sample = t_dense[idx]; x_sample = x_dense[idx]; y_sample = y_dense[idx]

    # --- Centrage pour stabilité numérique (optionnel mais sûr) ---
    x_mean, y_mean = float(np.mean(x_sample)), float(np.mean(y_sample))
    x_cent,  y_cent  = x_sample - x_mean, y_sample - y_mean


    # Lagrange/Newton : stabilisation optionnelle (affine) sur t
    if stabilize:
        tn, td = normalize_t(t_sample, t_dense)
        lag_x = lagrange(tn, x_cent)(td);    lag_y = lagrange(tn, y_cent)(td)
        newt_x = newton(tn, x_cent)(td);     newt_y = newton(tn, y_cent)(td)
    else:
        lag_x = lagrange(t_sample, x_cent)(t_dense);  lag_y = lagrange(t_sample, y_cent)(t_dense)
        newt_x = newton(t_sample, x_cent)(t_dense);   newt_y = newton(t_sample, y_cent)(t_dense)

    # --- Re-translation ---
    lag_x += x_mean; lag_y += y_mean
    newt_x += x_mean; newt_y += y_mean

    spl_x = CubicSpline(t_sample, x_sample, bc_type="natural")(t_dense)
    spl_y = CubicSpline(t_sample, y_sample, bc_type="natural")(t_dense)

    st.subheader("Trajectoire réelle vs interpolations")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_dense, y=y_dense, mode="lines", name="Trajectoire réelle", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=lag_x, y=lag_y, mode="lines", name="Lagrange", line=dict(color="red", dash="dash")))
    fig.add_trace(go.Scatter(x=newt_x, y=newt_y, mode="lines", name="Newton", line=dict(color="green", dash="dot")))
    fig.add_trace(go.Scatter(x=spl_x, y=spl_y, mode="lines", name="Spline cubique", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=x_sample, y=y_sample, mode="markers", name="Points échantillons", marker=dict(color="orange", size=7)))
    fig.update_layout(xaxis_title="x", yaxis_title="y")
    st.plotly_chart(fig, use_container_width=True)

    # Erreurs
    err_mae = {"Lagrange": 0.5*(mae(x_dense, lag_x) + mae(y_dense, lag_y)),
               "Newton":   0.5*(mae(x_dense, newt_x) + mae(y_dense, newt_y)),
               "Spline":   0.5*(mae(x_dense, spl_x)  + mae(y_dense, spl_y))}
    err_rmse = {"Lagrange": 0.5*(rmse(x_dense, lag_x) + rmse(y_dense, lag_y)),
                "Newton":   0.5*(rmse(x_dense, newt_x) + rmse(y_dense, newt_y)),
                "Spline":   0.5*(rmse(x_dense, spl_x)  + rmse(y_dense, spl_y))}
    st.subheader("Erreurs")
    col1, col2 = st.columns(2)
    with col1: st.write("**MAE**", {k: f"{v:.4e}" for k,v in err_mae.items()})
    with col2: st.write("**RMSE**", {k: f"{v:.4e}" for k,v in err_rmse.items()})

    # Détails math
    st.markdown("---")
    st.subheader("Détails mathématiques (formules du cours)")
    k = min(5, len(t_sample))
    st.markdown("**Lagrange (sur les k premiers points)**")
    st.latex(r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j=0, j \ne i}^{n} \frac{x - x_j}{x_i - x_j}")
    st.latex(pretty_polynomial(t_sample[:k], x_sample[:k], method='lagrange'))
    st.markdown("**Newton (différences divisées)**")
    st.latex(r"P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + \cdots + a_n\prod_{j=0}^{n-1}(x-x_j)")
    st.latex(pretty_polynomial(t_sample[:k], x_sample[:k], method='newton'))

# =========================
# MODE 2 : Trajet réel (carte/OSRM)
# =========================
else:
    if "map_key" not in st.session_state: st.session_state.map_key = 0
    if st.sidebar.button("🔁 Réinitialiser la carte"):
        st.session_state.map_key += 1
        st.experimental_rerun()

    st.markdown("**Pose deux marqueurs (crayon) pour Départ et Arrivée.**")
    m = folium.Map(location=[44.35, 2.57], zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    Draw(export=False, position="topleft",
         draw_options={"marker": True, "polyline": False, "polygon": False,
                       "rectangle": False, "circle": False, "circlemarker": False},
         edit_options={"edit": True, "remove": True}).add_to(m)
    map_state = st_folium(m, height=520, width=None, key=f"map_{st.session_state.map_key}",
                          returned_objects=["all_drawings", "last_clicked"])
    pts = extract_two_markers(map_state)

    # fallback clics
    if "click_pts" not in st.session_state: st.session_state.click_pts = []
    lc = (map_state or {}).get("last_clicked")
    if lc:
        pt = (lc["lat"], lc["lng"])
        if not st.session_state.click_pts or st.session_state.click_pts[-1] != pt:
            st.session_state.click_pts.append(pt); st.session_state.click_pts = st.session_state.click_pts[-2:]
    if len(pts) < 2 and len(st.session_state.click_pts) >= 2:
        pts = st.session_state.click_pts[-2:]

    if len(pts) < 2:
        st.info("Ajoute deux marqueurs (ou deux clics) pour construire un trajet.")
        st.stop()

    start, end = pts; st.success(f"Départ: {start} — Arrivée: {end}")

    try:
        route_latlon = get_route_osrm(start, end)
    except Exception as e:
        st.error(f"Erreur OSRM : {e}"); st.stop()
    if len(route_latlon) < 2:
        st.warning("Pas d’itinéraire exploitable. Déplace légèrement les points."); st.stop()

    route_map = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=13)
    folium.PolyLine([(la, lo) for la, lo in route_latlon], color="black", weight=4).add_to(route_map)
    folium.Marker(start, tooltip="Départ", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(end, tooltip="Arrivée", icon=folium.Icon(color="red")).add_to(route_map)
    st_folium(route_map, height=360, width=None, key=f"route_{st.session_state.map_key}_route")

    x_all, y_all = latlon_to_xy(route_latlon)
    t_all = arclength_param(x_all, y_all)
    # nettoyage doublons
    t_all, idxu = np.unique(t_all, return_index=True); x_all, y_all = x_all[idxu], y_all[idxu]
    if len(t_all) < 2:
        st.warning("Itinéraire trop court/plat. Déplace les points."); st.stop()

    # échantillonnage N points
    t_sample, x_sample, y_sample = resample_equidistant(t_all, x_all, y_all, n_sample)

    # --- Centrage + mise à l'échelle des sorties (x,y) pour stabilité numérique ---
    x_mean, y_mean = float(np.mean(x_sample)), float(np.mean(y_sample))
    x_cent, y_cent = x_sample - x_mean, y_sample - y_mean

    # échelle: amplitude max (évite division par 0)
    x_scale = float(np.max(np.abs(x_cent))) or 1.0
    y_scale = float(np.max(np.abs(y_cent))) or 1.0

    x_norm = x_cent / x_scale
    y_norm = y_cent / y_scale

    # spline naturelle = référence dense (évaluation sur t_dense commun)
    t_dense = np.linspace(t_sample.min(), t_sample.max(), 1200)
    csx = CubicSpline(t_sample, x_sample, bc_type="natural")
    csy = CubicSpline(t_sample, y_sample, bc_type="natural")
    x_ref, y_ref = csx(t_dense), csy(t_dense)  # référence (trajet reconstruit)

    # Lagrange/Newton — normalisation affine de t (option stabiliser)
    if stabilize:
        tn, td = normalize_t(t_sample, t_dense)
        lag_xn = lagrange(tn, x_norm)(td);    lag_yn = lagrange(tn, y_norm)(td)
        newt_xn = newton(tn, x_norm)(td);     newt_yn = newton(tn, y_norm)(td)
    else:
        lag_xn = lagrange(t_sample, x_norm)(t_dense);  lag_yn = lagrange(t_sample, y_norm)(t_dense)
        newt_xn = newton(t_sample, x_norm)(t_dense);   newt_yn = newton(t_sample, y_norm)(t_dense)

    # --- Re-dimensionnement + re-translation dans le repère originel ---
    lag_x = lag_xn * x_scale + x_mean
    lag_y = lag_yn * y_scale + y_mean
    newt_x = newt_xn * x_scale + x_mean
    newt_y = newt_yn * y_scale + y_mean


    # Tracé
    st.subheader("Trajectoire réelle vs interpolations")
    fig = go.Figure()

    # bornes calculées sur la route réelle et la spline, pas sur Lagrange/Newton
    xmin = float(min(x_all.min(), x_ref.min()))
    xmax = float(max(x_all.max(), x_ref.max()))
    ymin = float(min(y_all.min(), y_ref.min()))
    ymax = float(max(y_all.max(), y_ref.max()))
    # petite marge visuelle
    dx = 0.05 * (xmax - xmin + 1.0)
    dy = 0.05 * (ymax - ymin + 1.0)
    fig.update_xaxes(range=[xmin - dx, xmax + dx], title="x (m)")
    fig.update_yaxes(range=[ymin - dy, ymax + dy], title="y (m)")


    fig.add_trace(go.Scatter(x=x_all, y=y_all, mode="lines", name="Trajet OSRM (réel)", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode="lines", name="Spline cubique (réf)", line=dict(color="blue")))
    # Lagrange/Newton : on n’affiche que si valeurs finies
    if np.all(np.isfinite(lag_x)) and np.all(np.isfinite(lag_y)):
        fig.add_trace(go.Scatter(x=lag_x, y=lag_y, mode="lines", name="Lagrange", line=dict(color="red", dash="dash")))
    else:
        st.warning("Lagrange non tracé (instable avec cet ordre). Active « stabiliser » ou réduis N.")
    if np.all(np.isfinite(newt_x)) and np.all(np.isfinite(newt_y)):
        fig.add_trace(go.Scatter(x=newt_x, y=newt_y, mode="lines", name="Newton", line=dict(color="green", dash="dot")))
    else:
        st.warning("Newton non tracé (instable avec cet ordre). Active « stabiliser » ou réduis N.")
    fig.add_trace(go.Scatter(x=x_sample, y=y_sample, mode="markers", name=f"Échantillons (N={len(x_sample)})",
                             marker=dict(color="orange", size=6)))
    fig.update_layout(xaxis_title="x (m)", yaxis_title="y (m)", legend=dict(x=0, y=1))
    st.plotly_chart(fig, use_container_width=True)

    # Erreurs vs référence spline
    def err_pair(xr, yr, xa, ya):
        return 0.5*(mae(xr, xa) + mae(yr, ya)), 0.5*(rmse(xr, xa) + rmse(yr, ya))
    mae_l = rmse_l = mae_n = rmse_n = np.nan
    if np.all(np.isfinite(lag_x)) and np.all(np.isfinite(lag_y)):
        mae_l, rmse_l = err_pair(x_ref, y_ref, lag_x, lag_y)
    if np.all(np.isfinite(newt_x)) and np.all(np.isfinite(newt_y)):
        mae_n, rmse_n = err_pair(x_ref, y_ref, newt_x, newt_y)
    mae_s, rmse_s = err_pair(x_ref, y_ref, x_ref, y_ref)  # zéro
    st.subheader("Erreurs")
    col1, col2 = st.columns(2)
    with col1:
        st.write({"Lagrange": mae_l, "Newton": mae_n, "Spline (réf)": mae_s})
    with col2:
        st.write({"Lagrange": rmse_l, "Newton": rmse_n, "Spline (réf)": rmse_s})

    # Détails math (formules du cours)
    st.markdown("---")
    st.subheader("Détails mathématiques (formules du cours)")
    k = min(5, len(t_sample))
    st.markdown("**Lagrange (sur les k premiers points)**")
    st.latex(r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j=0, j \ne i}^{n} \frac{x - x_j}{x_i - x_j}")
    st.latex(pretty_polynomial(t_sample[:k], x_sample[:k], method='lagrange'))
    st.markdown("**Newton (différences divisées)**")
    st.latex(r"P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + \cdots + a_n\prod_{j=0}^{n-1}(x-x_j)")
    st.latex(pretty_polynomial(t_sample[:k], x_sample[:k], method='newton'))
