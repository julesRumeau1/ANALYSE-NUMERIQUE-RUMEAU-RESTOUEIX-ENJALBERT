import numpy as np
import requests
import streamlit as st
import plotly.graph_objects as go
import folium
import matplotlib.pyplot as plt
from streamlit_folium import st_folium
from folium.plugins import Draw
from scipy.interpolate import CubicSpline, PchipInterpolator
from pyproj import Transformer
from math_details import pretty_polynomial, latex_lagrange_basis, latex_newton_details


# ---------- utilitaires ----------
def mae(y_true, y_pred):  return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def compute_errors_pair(x_ref, y_ref, x_pred, y_pred):
    return 0.5 * (mae(x_ref, x_pred) + mae(y_ref, y_pred)), \
           0.5 * (rmse(x_ref, x_pred) + rmse(y_ref, y_pred))

def normalize_t(nodes, t_eval):
    a, b = float(nodes[0]), float(nodes[-1])
    if b == a:
        return np.zeros_like(nodes, float), np.zeros_like(t_eval, float)
    return (nodes - a) / (b - a), (t_eval - a) / (b - a)

def resample_equidistant(s, x, y, n):
    s_new = np.linspace(float(s[0]), float(s[-1]), n)
    fx, fy = PchipInterpolator(s, x), PchipInterpolator(s, y)
    return s_new, fx(s_new), fy(s_new)


# ---------- polynômes (cours) ----------
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


# ---------- app ----------
st.set_page_config(page_title="Interpolation polynomiale 2D", layout="wide")
st.title("Interpolation polynomiale 2D — Lagrange • Newton • Spline")

mode = st.sidebar.radio("Mode", ["Courbe synthétique", "Trajet réel (carte/OSRM)"])
n_sample = st.sidebar.slider("Nombre de points (échantillonnage des données)", 5, 200, 100, step=5)
stabilize = st.sidebar.checkbox("Stabiliser Lagrange/Newton (normaliser t)", value=True)


# ---------- synthétique ----------
def generate_ground_truth(n=1000):
    t = np.linspace(0, 10, n)
    x = np.cos(t) + 0.5 * np.sin(2 * t)
    y = np.sin(1.2 * t) + 0.3 * np.cos(3 * t)
    return t, x, y


# ---------- erreurs: courbes MAE/RMSE ----------
def _interp_all_on(t_sample, x_sample, y_sample, t_eval, stabilize=True):
    csx, csy = CubicSpline(t_sample, x_sample, bc_type="natural"), CubicSpline(t_sample, y_sample, bc_type="natural")
    spl_x, spl_y = csx(t_eval), csy(t_eval)

    xm, ym = float(np.mean(x_sample)), float(np.mean(y_sample))
    xc, yc = x_sample - xm, y_sample - ym
    xs, ys = float(np.max(np.abs(xc))) or 1.0, float(np.max(np.abs(yc))) or 1.0
    xn, yn = xc / xs, yc / ys

    if stabilize:
        tn, te = normalize_t(t_sample, t_eval)
        lag_xn, lag_yn = lagrange(tn, xn)(te), lagrange(tn, yn)(te)
        newt_xn, newt_yn = newton(tn, xn)(te), newton(tn, yn)(te)
    else:
        lag_xn, lag_yn = lagrange(t_sample, xn)(t_eval), lagrange(t_sample, yn)(t_eval)
        newt_xn, newt_yn = newton(t_sample, xn)(t_eval), newton(t_sample, yn)(t_eval)

    return (lag_xn * xs + xm, lag_yn * ys + ym), \
           (newt_xn * xs + xm, newt_yn * ys + ym), \
           (spl_x, spl_y)

def errors_vs_samples_SYNTH(min_n, max_n, t_dense, x_dense, y_dense, stabilize=True):
    out = {"n": [], "Lag_MAE": [], "Lag_RMSE": [], "New_MAE": [], "New_RMSE": [], "Spl_MAE": [], "Spl_RMSE": []}
    for n in range(min_n, max_n + 1):
        idx = np.linspace(0, len(t_dense) - 1, n, dtype=int)
        t_s, x_s, y_s = t_dense[idx], x_dense[idx], y_dense[idx]
        (lagx, lagy), (nwx, nwy), (spx, spy) = _interp_all_on(t_s, x_s, y_s, t_dense, stabilize=stabilize)
        mae_l, rmse_l = compute_errors_pair(x_dense, y_dense, lagx, lagy)
        mae_n, rmse_n = compute_errors_pair(x_dense, y_dense, nwx, nwy)
        mae_s, rmse_s = compute_errors_pair(x_dense, y_dense, spx, spy)
        out["n"].append(n)
        out["Lag_MAE"].append(mae_l); out["Lag_RMSE"].append(rmse_l)
        out["New_MAE"].append(mae_n); out["New_RMSE"].append(rmse_n)
        out["Spl_MAE"].append(mae_s); out["Spl_RMSE"].append(rmse_s)
    return out

def errors_vs_samples_REAL(min_n, max_n, t_all, x_all, y_all, t_eval, x_ref, y_ref, stabilize=True):
    out = {"n": [], "Lag_MAE": [], "Lag_RMSE": [], "New_MAE": [], "New_RMSE": [], "Spl_MAE": [], "Spl_RMSE": []}
    for n in range(min_n, max_n + 1):
        t_s, x_s, y_s = resample_equidistant(t_all, x_all, y_all, n)
        (lagx, lagy), (nwx, nwy), (spx, spy) = _interp_all_on(t_s, x_s, y_s, t_eval, stabilize=stabilize)
        mae_l, rmse_l = compute_errors_pair(x_ref, y_ref, lagx, lagy)
        mae_n, rmse_n = compute_errors_pair(x_ref, y_ref, nwx, nwy)
        mae_s, rmse_s = compute_errors_pair(x_ref, y_ref, spx, spy)
        out["n"].append(n)
        out["Lag_MAE"].append(mae_l); out["Lag_RMSE"].append(rmse_l)
        out["New_MAE"].append(mae_n); out["New_RMSE"].append(rmse_n)
        out["Spl_MAE"].append(mae_s); out["Spl_RMSE"].append(rmse_s)
    return out

def show_error_curves(errors_dict):
    n = errors_dict["n"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=130)
    ax[0].plot(n, errors_dict["Lag_MAE"], "r--", label="Lagrange")
    ax[0].plot(n, errors_dict["New_MAE"], "g-.", label="Newton")
    ax[0].plot(n, errors_dict["Spl_MAE"], "b-", label="Spline cubique")
    ax[0].set_title("Évolution de la MAE"); ax[0].set_xlabel("Nombre de points d'échantillonnage"); ax[0].set_ylabel("MAE")
    ax[0].grid(True); ax[0].legend()
    ax[1].plot(n, errors_dict["Lag_RMSE"], "r--", label="Lagrange")
    ax[1].plot(n, errors_dict["New_RMSE"], "g-.", label="Newton")
    ax[1].plot(n, errors_dict["Spl_RMSE"], "b-", label="Spline cubique")
    ax[1].set_title("Évolution de la RMSE"); ax[1].set_xlabel("Nombre de points d'échantillonnage"); ax[1].set_ylabel("RMSE")
    ax[1].grid(True); ax[1].legend()
    plt.tight_layout()
    st.pyplot(fig)


# ---------- carte / outils réels ----------
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)

def get_route_osrm(start, end):
    url = f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full&geometries=geojson"
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
    x, y = _transformer.transform(lons, lats)
    return np.asarray(x, float), np.asarray(y, float)

def arclength_param(x, y):
    ds = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], np.cumsum(ds)])

def extract_two_markers(map_state):
    pts, drawings = [], (map_state.get("all_drawings") or []) if map_state else []
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


# ================== MODE 1 : Courbe synthétique ==================
if mode == "Courbe synthétique":
    t_dense, x_dense, y_dense = generate_ground_truth(n=1000)
    idx = np.linspace(0, len(t_dense) - 1, n_sample, dtype=int)
    t_sample, x_sample, y_sample = t_dense[idx], x_dense[idx], y_dense[idx]

    xm, ym = float(np.mean(x_sample)), float(np.mean(y_sample))
    x_cent, y_cent = x_sample - xm, y_sample - ym

    if stabilize:
        tn, td = normalize_t(t_sample, t_dense)
        lag_x, lag_y = lagrange(tn, x_cent)(td), lagrange(tn, y_cent)(td)
        newt_x, newt_y = newton(tn, x_cent)(td), newton(tn, y_cent)(td)
    else:
        lag_x, lag_y = lagrange(t_sample, x_cent)(t_dense), lagrange(t_sample, y_cent)(t_dense)
        newt_x, newt_y = newton(t_sample, x_cent)(t_dense), newton(t_sample, y_cent)(t_dense)

    lag_x += xm; lag_y += ym
    newt_x += xm; newt_y += ym

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

    mae_l, rmse_l = compute_errors_pair(x_dense, y_dense, lag_x, lag_y)
    mae_n, rmse_n = compute_errors_pair(x_dense, y_dense, newt_x, newt_y)
    mae_s, rmse_s = compute_errors_pair(x_dense, y_dense, spl_x, spl_y)

    st.subheader("Erreurs")
    c1, c2 = st.columns(2)
    with c1: st.write({"Lagrange": mae_l, "Newton": mae_n, "Spline (réf)": mae_s})
    with c2: st.write({"Lagrange": rmse_l, "Newton": rmse_n, "Spline (réf)": rmse_s})

    with st.expander("📈 Évolution MAE/RMSE quand on fait varier le nombre de points"):
        errs = errors_vs_samples_SYNTH(10, min(70, max(10, n_sample)), t_dense, x_dense, y_dense, stabilize=stabilize)
        show_error_curves(errs)

    st.markdown("---"); st.subheader("Détails mathématiques (formules du cours)")
    m_show = min(5, len(t_sample))
    st.markdown("**Lagrange — bases \\(\\ell_0,\\ldots,\\ell_4\\) et polynôme**")
    st.latex(r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j=0, j \ne i}^{n} \frac{x - x_j}{x_i - x_j}")
    for li in latex_lagrange_basis(t_sample[:m_show], m=m_show, var="x"): st.latex(li)
    st.latex(pretty_polynomial(t_sample[:m_show], x_sample[:m_show], method='lagrange'))
    st.markdown("**Newton — différences divisées**")
    st.latex(r"P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + \cdots + a_n\prod_{j=0}^{n-1}(x-x_j)")
    a_lines, newton_poly = latex_newton_details(t_sample[:m_show], x_sample[:m_show], m=m_show, var="x")
    for line in a_lines: st.latex(line)
    st.latex(newton_poly)
    st.markdown("**Forme développée du polynôme de Newton**")
    st.latex(pretty_polynomial(t_sample[:m_show], x_sample[:m_show], method='newton'))


# ================== MODE 2 : Trajet réel (carte/OSRM) ==================
else:
    if "map_key" not in st.session_state: st.session_state.map_key = 0
    if st.sidebar.button("🔁 Réinitialiser la carte"):
        st.session_state.map_key += 1
        st.experimental_rerun()

    st.markdown("**Pose deux marqueurs (crayon) pour Départ et Arrivée.**")
    m = folium.Map(location=[44.35, 2.57], zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    Draw(export=False, position="topleft",
         draw_options={"marker": True, "polyline": False, "polygon": False, "rectangle": False, "circle": False, "circlemarker": False},
         edit_options={"edit": True, "remove": True}).add_to(m)
    map_state = st_folium(m, height=520, width=None, key=f"map_{st.session_state.map_key}",
                          returned_objects=["all_drawings", "last_clicked"])
    pts = extract_two_markers(map_state)

    if "click_pts" not in st.session_state: st.session_state.click_pts = []
    lc = (map_state or {}).get("last_clicked")
    if lc:
        pt = (lc["lat"], lc["lng"])
        if not st.session_state.click_pts or st.session_state.click_pts[-1] != pt:
            st.session_state.click_pts.append(pt)
            st.session_state.click_pts = st.session_state.click_pts[-2:]
    if len(pts) < 2 and len(st.session_state.click_pts) >= 2: pts = st.session_state.click_pts[-2:]
    if len(pts) < 2:
        st.info("Ajoute deux marqueurs (ou deux clics) pour construire un trajet."); st.stop()

    start, end = pts; st.success(f"Départ: {start} — Arrivée: {end}")

    try:
        route_latlon = get_route_osrm(start, end)
    except Exception as e:
        st.error(f"Erreur OSRM : {e}"); st.stop()
    if len(route_latlon) < 2:
        st.warning("Pas d’itinéraire exploitable. Déplace légèrement les points."); st.stop()

    route_map = folium.Map(location=[(start[0] + end[0]) / 2, (start[1] + end[1]) / 2], zoom_start=13)
    folium.PolyLine([(la, lo) for la, lo in route_latlon], color="black", weight=4).add_to(route_map)
    folium.Marker(start, tooltip="Départ", icon=folium.Icon(color="green")).add_to(route_map)
    folium.Marker(end, tooltip="Arrivée", icon=folium.Icon(color="red")).add_to(route_map)
    st_folium(route_map, height=360, width=None, key=f"route_{st.session_state.map_key}_route")

    x_all, y_all = latlon_to_xy(route_latlon)
    t_all = arclength_param(x_all, y_all)
    t_all, idxu = np.unique(t_all, return_index=True)
    x_all, y_all = x_all[idxu], y_all[idxu]
    if len(t_all) < 2: st.warning("Itinéraire trop court/plat. Déplace les points."); st.stop()

    t_sample, x_sample, y_sample = resample_equidistant(t_all, x_all, y_all, n_sample)

    xm, ym = float(np.mean(x_sample)), float(np.mean(y_sample))
    x_cent, y_cent = x_sample - xm, y_sample - ym
    xs, ys = float(np.max(np.abs(x_cent))) or 1.0, float(np.max(np.abs(y_cent))) or 1.0
    x_norm, y_norm = x_cent / xs, y_cent / ys

    t_dense = np.linspace(t_sample.min(), t_sample.max(), 1200)
    csx, csy = CubicSpline(t_sample, x_sample, bc_type="natural"), CubicSpline(t_sample, y_sample, bc_type="natural")
    x_ref, y_ref = csx(t_dense), csy(t_dense)

    if stabilize:
        tn, td = normalize_t(t_sample, t_dense)
        lag_xn, lag_yn = lagrange(tn, x_norm)(td), lagrange(tn, y_norm)(td)
        newt_xn, newt_yn = newton(tn, x_norm)(td), newton(tn, y_norm)(td)
    else:
        lag_xn, lag_yn = lagrange(t_sample, x_norm)(t_dense), lagrange(t_sample, y_norm)(t_dense)
        newt_xn, newt_yn = newton(t_sample, x_norm)(t_dense), newton(t_sample, y_norm)(t_dense)

    lag_x, lag_y = lag_xn * xs + xm, lag_yn * ys + ym
    newt_x, newt_y = newt_xn * xs + xm, newt_yn * ys + ym

    st.subheader("Trajectoire réelle vs interpolations")
    fig = go.Figure()

    xmin, xmax = float(min(x_all.min(), x_ref.min())), float(max(x_all.max(), x_ref.max()))
    ymin, ymax = float(min(y_all.min(), y_ref.min())), float(max(y_all.max(), y_ref.max()))
    dx, dy = 0.05 * (xmax - xmin + 1.0), 0.05 * (ymax - ymin + 1.0)
    fig.update_xaxes(range=[xmin - dx, xmax + dx], title="x (m)")
    fig.update_yaxes(range=[ymin - dy, ymax + dy], title="y (m)")

    fig.add_trace(go.Scatter(x=x_all, y=y_all, mode="lines", name="Trajet OSRM (réel)", line=dict(color="black", width=3)))
    fig.add_trace(go.Scatter(x=x_ref, y=y_ref, mode="lines", name="Spline cubique (réf)", line=dict(color="blue")))
    if np.all(np.isfinite(lag_x)) and np.all(np.isfinite(lag_y)):
        fig.add_trace(go.Scatter(x=lag_x, y=lag_y, mode="lines", name="Lagrange", line=dict(color="red", dash="dash")))
    else:
        st.warning("Lagrange non tracé (instable).")
    if np.all(np.isfinite(newt_x)) and np.all(np.isfinite(newt_y)):
        fig.add_trace(go.Scatter(x=newt_x, y=newt_y, mode="lines", name="Newton", line=dict(color="green", dash="dot")))
    else:
        st.warning("Newton non tracé (instable).")
    fig.add_trace(go.Scatter(x=x_sample, y=y_sample, mode="markers", name=f"Échantillons (N={len(x_sample)})",
                             marker=dict(color="orange", size=6)))
    fig.update_layout(legend=dict(x=0, y=1))
    st.plotly_chart(fig, use_container_width=True)

    mae_l = rmse_l = mae_n = rmse_n = np.nan
    if np.all(np.isfinite(lag_x)) and np.all(np.isfinite(lag_y)):
        mae_l, rmse_l = compute_errors_pair(x_ref, y_ref, lag_x, lag_y)
    if np.all(np.isfinite(newt_x)) and np.all(np.isfinite(newt_y)):
        mae_n, rmse_n = compute_errors_pair(x_ref, y_ref, newt_x, newt_y)
    mae_s, rmse_s = compute_errors_pair(x_ref, y_ref, x_ref, y_ref)

    st.subheader("Erreurs")
    c1, c2 = st.columns(2)
    with c1: st.write({"Lagrange": mae_l, "Newton": mae_n, "Spline (réf)": mae_s})
    with c2: st.write({"Lagrange": rmse_l, "Newton": rmse_n, "Spline (réf)": rmse_s})

    with st.expander("📈 Évolution MAE/RMSE quand on fait varier le nombre de points"):
        errs = errors_vs_samples_REAL(10, min(70, int(len(t_all))), t_all, x_all, y_all, t_dense, x_ref, y_ref, stabilize=stabilize)
        show_error_curves(errs)

    st.markdown("---"); st.subheader("Détails mathématiques (formules du cours)")
    m_show = min(5, len(t_sample))
    st.markdown("**Lagrange — bases \\(\\ell_0,\\ldots,\\ell_4\\) et polynôme**")
    st.latex(r"P_n(x) = \sum_{i=0}^{n} y_i \prod_{j=0, j \ne i}^{n} \frac{x - x_j}{x_i - x_j}")
    for li in latex_lagrange_basis(t_sample[:m_show], m=m_show, var="x"): st.latex(li)
    st.latex(pretty_polynomial(t_sample[:m_show], x_sample[:m_show], method='lagrange'))
    st.markdown("**Newton — différences divisées**")
    st.latex(r"P_n(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + \cdots + a_n\prod_{j=0}^{n-1}(x-x_j)")
    a_lines, newton_poly = latex_newton_details(t_sample[:m_show], x_sample[:m_show], m=m_show, var="x")
    for line in a_lines: st.latex(line)
    st.latex(newton_poly)
    st.markdown("**Forme développée du polynôme de Newton**")
    st.latex(pretty_polynomial(t_sample[:m_show], x_sample[:m_show], method='newton'))
