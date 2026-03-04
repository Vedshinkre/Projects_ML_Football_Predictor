"""
Premier League Match Prediction Interface.

This Streamlit application serves as the front-end for our Level 6 (Ultimate) 
Random Forest model. It loads the pre-trained artifacts (model, scaler, Elo/H2H memory) 
from the local disk, fetches real-time match data for form calculations, and outputs 
a probability matrix for upcoming fixtures.
"""

import streamlit as st
import pandas as pd
import joblib
import requests
import io
import numpy as np
from datetime import datetime, time
from config import MODEL_SAVE_PATH, FEATURE_SETS, DATA_URL

# ---------------------------------------------------------
# 1. SETUP & CACHING
# ---------------------------------------------------------
st.set_page_config(page_title="Premier League Predictor", layout="centered")

@st.cache_resource
def load_brain():
    """Loads the serialized model pipelines and historical state dictionaries."""
    try:
        model = joblib.load(f'{MODEL_SAVE_PATH}/final_model.pkl')
        team_map = joblib.load(f'{MODEL_SAVE_PATH}/team_map.pkl')
        elo_dict = joblib.load(f'{MODEL_SAVE_PATH}/elo_dict.pkl')
        h2h_dict = joblib.load(f'{MODEL_SAVE_PATH}/h2h_dict.pkl')
        return model, team_map, elo_dict, h2h_dict
    except Exception as e:
        return None, None, None, None

@st.cache_data
def load_fifa_data():
    """Loads and standardizes EA Sports FIFA squad valuations."""
    try:
        df = pd.read_csv('data/raw/fifa_ratings_cleaned.csv')
        df = df.sort_values('Season_Key', ascending=False).drop_duplicates('Team')
        return df.set_index('Team')
    except Exception:
        return None

@st.cache_data(ttl=3600) 
def fetch_live_data():
    """Downloads the latest season match data for dynamic form calculations."""
    try:
        s = requests.get(DATA_URL).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.sort_values('Date')
        return df
    except Exception:
        return None

def get_stats(team_name, df):
    """Calculates 3-game rolling averages for goals, shots, and set-pieces."""
    team_games = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)].copy()
    if len(team_games) < 3: return None
    last_3 = team_games.tail(3)
    
    stats = {"GF": [], "GA": [], "Sh": [], "SoT": [], "Corn": []}
    for _, row in last_3.iterrows():
        if row['HomeTeam'] == team_name:
            stats["GF"].append(row['FTHG']); stats["GA"].append(row['FTAG'])
            stats["Sh"].append(row['HS']); stats["SoT"].append(row['HST'])
            stats["Corn"].append(row['HC'])
        else:
            stats["GF"].append(row['FTAG']); stats["GA"].append(row['FTHG'])
            stats["Sh"].append(row['AS']); stats["SoT"].append(row['AST'])
            stats["Corn"].append(row['AC'])
            
    return {k: np.mean(v) for k, v in stats.items()}

# ---------------------------------------------------------
# 2. UI & INFERENCE LOGIC
# ---------------------------------------------------------
st.title("Premier League Prediction Engine")
st.caption("Model Architecture: Level 6 Random Forest (Elo, Form, Matchups, Squad Valuation)")

# Initialize Assets
model, team_map, base_elo, base_h2h = load_brain()
fifa_data = load_fifa_data()
live_data = fetch_live_data()

if model is None:
    st.error("System Error: Pre-trained artifacts not found. Please execute src/train.py.")
    st.stop()

# --- INPUT PARAMETERS ---
st.markdown("### Match Parameters")
col1, col2 = st.columns(2)
teams = sorted([t for t in base_elo.keys() if isinstance(t, str)])

with col1: 
    default_home = "Liverpool" if "Liverpool" in teams else teams[0]
    home = st.selectbox("Home Team", teams, index=teams.index(default_home))
with col2: 
    default_away = "Man City" if "Man City" in teams else teams[1]
    away = st.selectbox("Away Team", teams, index=teams.index(default_away))

c3, c4 = st.columns(2)
with c3: match_date = st.date_input("Match Date", datetime.now())
with c4: match_time = st.time_input("Kickoff Time", value=time(15, 0))

# --- EXECUTE PREDICTION ---
if st.button("Generate Prediction", type="primary", use_container_width=True):
    if home == away:
        st.warning("Invalid configuration: Teams must be distinct.")
    else:
        # Extract live form
        h_stats = get_stats(home, live_data)
        a_stats = get_stats(away, live_data)
        
        if not h_stats or not a_stats:
            st.error("Insufficient live match data to compute short-term form.")
        else:
            # Extract historical squad valuations
            try:
                h_fifa = fifa_data.loc[home]
                a_fifa = fifa_data.loc[away]
            except KeyError:
                st.info("Notice: Team missing from FIFA database. Utilizing league average estimates.")
                h_fifa = fifa_data.loc[home] if home in fifa_data.index else pd.Series({'overall': 75, 'attack': 75, 'midfield': 75, 'defence': 75, 'club_worth_eur': 150.0, 'starting_xi_average_age': 26})
                a_fifa = fifa_data.loc[away] if away in fifa_data.index else pd.Series({'overall': 75, 'attack': 75, 'midfield': 75, 'defence': 75, 'club_worth_eur': 150.0, 'starting_xi_average_age': 26})

            # --- RENDER MATCH CONTEXT ---
            st.markdown("### Recent Form")
            
            col_home, col_vs, col_away = st.columns([2, 1, 2])
            
            with col_home:
                st.markdown(f"#### 🏠 {home}")
                st.metric("FIFA Squad Rating", f"{int(h_fifa['overall'])}")
                st.metric("Goals Scored (Last 3)", f"{h_stats['GF']:.1f} / game")
                st.metric("Goals Conceded (Last 3)", f"{h_stats['GA']:.1f} / game")
                
            with col_vs:
                st.markdown("<h1 style='text-align: center; margin-top: 60px; color: #888888;'>VS</h1>", unsafe_allow_html=True)
                
            with col_away:
                st.markdown(f"#### ✈️ {away}")
                st.metric("FIFA Squad Rating", f"{int(a_fifa['overall'])}")
                st.metric("Goals Scored (Last 3)", f"{a_stats['GF']:.1f} / game")
                st.metric("Goals Conceded (Last 3)", f"{a_stats['GA']:.1f} / game")
                
            st.divider()

            # --- COMPUTE FEATURE VECTOR ---
            elo_diff = base_elo.get(home, 1500) - base_elo.get(away, 1500)
            
            pair_key = (home, away)
            history = base_h2h.get(pair_key, [])
            h2h_rate = sum(history) / len(history) if len(history) > 0 else 0.5
            
            h_xgd = (h_stats['SoT']/4.0) - (a_stats['SoT']/4.0)
            
            ov_diff = h_fifa['overall'] - a_fifa['overall']
            att_diff = h_fifa['attack'] - a_fifa['attack']
            mid_diff = h_fifa['midfield'] - a_fifa['midfield']
            def_diff = h_fifa['defence'] - a_fifa['defence']
            worth_diff = np.log1p(h_fifa['club_worth_eur']) - np.log1p(a_fifa['club_worth_eur']) 
            age_diff = h_fifa['starting_xi_average_age'] - a_fifa['starting_xi_average_age']

            row = [
                1,                      # Venue_code
                team_map.get(away, 0),  # Opp_code
                match_time.hour,        # Hour
                match_date.weekday(),   # Day_code
                elo_diff,
                h2h_rate,
                h_xgd,
                ov_diff, att_diff, mid_diff, def_diff, worth_diff, age_diff,
                h_stats['GF'], h_stats['GA'], h_stats['Sh'], h_stats['SoT'], h_stats['Corn'],
                a_stats['GF'], a_stats['GA'], a_stats['Sh'], a_stats['SoT']
            ]
            
            features = FEATURE_SETS["Level 6 (Ultimate)"]
            
            # --- MODEL INFERENCE ---
            if len(row) != len(features):
                st.error(f"Architecture Mismatch: Expected {len(features)} features, generated {len(row)}.")
            else:
                input_df = pd.DataFrame([row], columns=features)
                prob = model.predict_proba(input_df)[0][1]
                
                st.subheader("Model Projection")
                
                if prob > 0.55:
                    st.success(f"Projected Outcome: {home} Win")
                    st.progress(prob, text=f"Statistical Confidence: {prob:.1%}")
                elif prob < 0.45:
                    st.error(f"Projected Outcome: {away} Win or Draw")
                    st.progress(1-prob, text=f"Statistical Confidence: {1-prob:.1%}")
                else:
                    st.warning("Projected Outcome: Marginal / Too Close to Call")
                    st.info(f"Home Win Probability: {prob:.1%} (No statistical edge detected)")