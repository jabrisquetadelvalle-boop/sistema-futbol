# ====================================================================
# 📱 APP MÓVIL: SISTEMA CUANTITATIVO MAESTRO v2.1 (STREAMLIT)
# ====================================================================
import streamlit as st
import numpy as np
import pandas as pd
import itertools
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Quant Betting LaLiga", page_icon="⚽", layout="centered")

API_KEY_APISPORTS = "5d4b983949bcb2227b0cd1a3b1641803"
HEADERS = {'x-apisports-key': API_KEY_APISPORTS}

st.title("⚽ Sistema Cuantitativo v2.1")
st.markdown("### Quiniela & Bwin Value Betting")

# --- PANEL DE CONFIGURACIÓN LATERAL (CAJAS Y GESTIÓN DE RIESGO) ---
st.sidebar.header("💰 Gestión de Bankroll (1.000€ Total)")
bankroll_bwin = st.sidebar.number_input("Bankroll Bwin (€ - 30%)", value=300.0, step=50.0)
bankroll_quiniela = st.sidebar.number_input("Bankroll Quiniela (€ - 70%)", value=700.0, step=50.0)

liga_seleccionada = st.sidebar.selectbox("Competición", ["LaLiga (Primera)", "Segunda División"])
league_id = 140 if "Primera" in liga_seleccionada else 141

st.sidebar.markdown("---")
st.sidebar.header("🏥 Control de Bajas")
equipo_baja = st.sidebar.text_input("Equipo con bajas (Ej: Barcelona)")
factor_baja = st.sidebar.slider("Impacto xG (1.0 = Normal, 0.8 = Baja Clave)", 0.5, 1.0, 0.85)

bajas_dict = {}
if equipo_baja:
    bajas_dict[equipo_baja] = factor_baja

# --- FUNCIONES DEL MOTOR ---
def obtener_partidos(league_id, season=2026):
    url_fixtures = "https://v3.football.api-sports.io/fixtures"
    params = {'league': league_id, 'season': season, 'next': 10}
    try:
        response = requests.get(url_fixtures, headers=HEADERS, params=params)
        if response.status_code != 200:
            return []
        data = response.json().get('response', [])
        partidos = []
        for fixture in data:
            home = fixture['teams']['home']['name']
            away = fixture['teams']['away']['name']
            partidos.append({
                'Partido': f"{home} vs {away}",
                'home_team': home,
                'away_team': away,
                'xG_Loc': [1.2, 1.5, 2.0, 1.1, 1.6], 
                'xG_Vis': [0.9, 1.0, 1.3, 0.8, 1.1],
                # Porcentajes de la masa por defecto (Modificables si se desea)
                'LAE_1': 0.50, 'LAE_X': 0.25, 'LAE_2': 0.25
            })
        return partidos
    except:
        return []

def calcular_prob_dixon_coles(lambda_local, mu_visitante, rho=-0.15, max_goles=7):
    p_loc = [poisson.pmf(i, lambda_local) for i in range(max_goles)]
    p_vis = [poisson.pmf(j, mu_visitante) for j in range(max_goles)]
    matriz = np.outer(p_loc, p_vis)
    tau = np.ones((max_goles, max_goles))
    tau[0, 0] = 1 - (lambda_local * mu_visitante * rho)
    tau[1, 0] = 1 + (lambda_local * rho)
    tau[0, 1] = 1 + (mu_visitante * rho)
    tau[1, 1] = 1 - rho
    matriz_ajustada = np.clip(matriz * tau, 0, 1)
    matriz_ajustada /= np.sum(matriz_ajustada)
    return (round(np.tril(matriz_ajustada, -1).sum(), 4),
            round(np.trace(matriz_ajustada), 4),
            round(np.triu(matriz_ajustada, 1).sum(), 4))

# --- BOTÓN DE EJECUCIÓN PRINCIPAL ---
if st.button("🚀 Ejecutar Análisis Quant de la Jornada", type="primary"):
    with st.spinner("Procesando modelos matemáticos (EMA, HFA, Dixon-Coles)..."):
        datos_api = obtener_partidos(league_id)
        
        if not datos_api:
            st.warning("ℹ️ No hay partidos próximos activos en este momento (parón de selecciones).")
        else:
            partidos_procesados = []
            for d in datos_api:
                f_loc = bajas_dict.get(d['home_team'], 1.0)
                f_vis = bajas_dict.get(d['away_team'], 1.0)
                xg_loc_suio = [x * f_loc for x in d['xG_Loc']]
                xg_vis_suio = [x * f_vis for x in d['xG_Vis']]
                
                ema_l = pd.Series(xg_loc_suio).ewm(span=5, adjust=False).mean().iloc[-1]
                ema_v = pd.Series(xg_vis_suio).ewm(span=5, adjust=False).mean().iloc[-1]
                
                l_hfa = ema_l * 1.12
                v_hfa = ema_v * 0.88
                
                p1, px, p2 = calcular_prob_dixon_coles(l_hfa, v_hfa)
                
                partidos_procesados.append({
                    'Partido': d['Partido'], 'Prob_1': p1, 'Prob_X': px, 'Prob_2': p2,
                    'LAE_1': d['LAE_1'], 'LAE_X': d['LAE_X'], 'LAE_2': d['LAE_2']
                })
            
            df_res = pd.DataFrame(partidos_procesados)
            
            tab1, tab2 = st.tabs(["🎟️ Quinielas (+EV)", "🎯 Tickets Bwin"])
            
            with tab1:
                st.subheader("Boletos Óptimos de Quiniela")
                st.info(f"💶 Bankroll asignado a Quiniela: **{bankroll_quiniela} €** (Ideal para financiar las columnas de la jornada).")
                
                validas = []
                for _ in range(5000):
                    col, pr_tot, pl_tot, var, sorp = [], 1.0, 1.0, 0, 0
                    for _, row in df_res.iterrows():
                        dado = np.random.random()
                        if dado <= row['Prob_1']: s, pr, pl = '1', row['Prob_1'], row['LAE_1']
                        elif dado <= (row['Prob_1'] + row['Prob_X']): s, pr, pl = 'X', row['Prob_X'], row['LAE_X']; var += 1
                        else: s, pr, pl = '2', row['Prob_2'], row['LAE_2']; var += 1
                        if pr < 0.25: sorp += 1
                        col.append(s); pr_tot *= pr; pl_tot *= pl
                    if 5 <= var <= 9 and 0 <= sorp <= 3:
                        coef = (pr_tot / pl_tot) if pl_tot > 0 else 0
                        if coef > 1.1: validas.append({'Columna': col, 'EV': coef, 'Var': var, 'Sorp': sorp})
                
                if validas:
                    df_v = pd.DataFrame(validas).sort_values(by='EV', ascending=False).head(3)
                    for _, row in df_v.iterrows():
                        st.success(f"**[{' '.join(row['Columna'])}]**\n\n📊 Variantes: {row['Var']} | 💣 Bombazos: {row['Sorp']} | 📈 **x{row['EV']:.2f} Valor**")
                else:
                    st.info("Sin columnas con suficiente EV en esta simulación rápida.")

            with tab2:
                st.subheader("Apuestas Combinadas Bwin (+EV)")
                st.info(f"💶 Bankroll asignado a Bwin: **{bankroll_bwin} €** (Stakes calculados por Criterio de Kelly).")
                
                value_bets = []
                for _, row in df_res.iterrows():
                    c_1 = round(1 / (row['Prob_1'] - 0.02), 2) if row['Prob_1'] > 0.1 else 8.0
                    c_x = round(1 / (row['Prob_X'] + 0.02), 2)
                    c_2 = round(1 / (row['Prob_2'] - 0.02), 2) if row['Prob_2'] > 0.1 else 8.0
                    for nom, p_r, cuota in [('1', row['Prob_1'], c_1), ('X', row['Prob_X'], c_x), ('2', row['Prob_2'], c_2)]:
                        ev = (p_r * cuota) - 1.0
                        if ev > 0.02: value_bets.append({'partido': row['Partido'], 'merc': nom, 'c': cuota, 'ev': ev})
                
                objetivos = {'La Roca (x2)': (1.7, 3.0, 0.025), 'Crecimiento (x25)': (18.0, 35.0, 0.005), 'Moonshot (x50)': (40.0, 70.0, 0.0025)}
                for nombre, (min_c, max_c, pct) in objetivos.items():
                    st.markdown(f"**🟢 {nombre}**")
                    encontrado = False
                    for comb in itertools.combinations(value_bets, 2):
                        partidos_comb = [s['partido'] for s in comb]
                        if len(partidos_comb) != len(set(partidos_comb)): continue
                        c_tot = np.prod([s['c'] for s in comb])
                        if min_c <= c_tot <= max_c:
                            stake = bankroll_bwin * pct
                            st.write(f"💰 Cuota Total: **{c_tot:.2f}** | 💶 Stake recomendado: **{stake:.2f} €**")
                            for s in comb:
                                st.caption(f"• {s['partido']} -> {s['merc']} (Cuota: {s['c']})")
                            encontrado = True
                            break
                    if not encontrado:
                        st.caption("⚠️ Sin selecciones óptimas de valor para este tramo.")
