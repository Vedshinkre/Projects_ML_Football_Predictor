"""
Production Model Training Pipeline.

This script executes the final training run for the Level 6 (Ultimate) Random Forest model
using the optimal hyperparameters discovered during Grid Search. 

Crucially, it also recalculates and exports the final state of the dynamic features 
(Elo Ratings and Head-to-Head records). These dictionaries act as the AI's "memory", 
allowing the web application to instantly look up a team's current strength without 
re-processing 10 years of historical data.
"""

import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Local project imports
from config import TRAIN_DATA_PATH, MODEL_SAVE_PATH, FEATURE_SETS
from feature_engineering import prepare_data 

def get_final_state(df):
    """
    Simulates the historical timeline to capture the exact Elo and Head-to-Head 
    state at the very end of the training dataset.
    """
    # 1. Re-calculate Elo to get the final dictionary state
    current_elo = {team: 1500 for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()}
    k_factor = 20
    
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        r_home, r_away = current_elo.get(home, 1500), current_elo.get(away, 1500)
        
        prob_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        actual = 1 if row['FTR'] == 'H' else 0.5 if row['FTR'] == 'D' else 0
        
        current_elo[home] = r_home + k_factor * (actual - prob_home)
        current_elo[away] = r_away + k_factor * ((1 - actual) - (1 - prob_home))

    # 2. Re-calculate Head-to-Head to get the final historical matrix
    h2h_history = {}
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        pair_key = (home, away)
        
        if row['FTR'] == 'H':
            res_home, res_away = 1, 0
        elif row['FTR'] == 'D':
            res_home, res_away = 0.5, 0.5
        else:
            res_home, res_away = 0, 1
            
        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append(res_home)
        
        rev_key = (away, home)
        if rev_key not in h2h_history: h2h_history[rev_key] = []
        h2h_history[rev_key].append(res_away)

    return current_elo, h2h_history

def train_final_model():
    print("STARTING FINAL TRAINING: LEVEL 6 (ULTIMATE)")
    
    # ---------------------------------------------------------
    # 1. Load & Prepare Data
    # ---------------------------------------------------------
    print(" -> Loading historical dataset...")
    raw_df = pd.read_csv(TRAIN_DATA_PATH)
    
    # Run the standard feature engineering pipeline
    full_df, team_map = prepare_data(raw_df)
    
    # ---------------------------------------------------------
    # 2. Capture the "Memory" (Elo & H2H State)
    # ---------------------------------------------------------
    raw_sorted = raw_df.copy()
    raw_sorted['Date'] = pd.to_datetime(raw_sorted['Date'], dayfirst=True, errors='coerce')
    raw_sorted = raw_sorted.sort_values('Date')
    
    print(" -> Capturing final Elo & H2H state for production inference...")
    final_elo, final_h2h = get_final_state(raw_sorted)
    
    # ---------------------------------------------------------
    # 3. Select Features
    # ---------------------------------------------------------
    features = FEATURE_SETS["Level 6 (Ultimate)"]
    print(f" -> Utilizing {len(features)} Features (FIFA Class + Form + Elo)")
    
    # Verify all configured columns exist in the processed dataframe
    missing = [c for c in features if c not in full_df.columns]
    if missing:
        print(f"Error: Missing columns {missing}")
        return

    # Drop NaNs to ensure stable training for the Random Forest
    full_df = full_df.dropna(subset=features)
    X = full_df[features]
    y = full_df["Target"]
    
    # ---------------------------------------------------------
    # 4. Train the Production Model
    # ---------------------------------------------------------
    print(" -> Training Tuned Random Forest (Level 6)...")
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('clf', RandomForestClassifier(
            n_estimators=300,        
            max_depth=5,             
            min_samples_split=20,    
            min_samples_leaf=1,      
            max_features='log2',     
            random_state=42
        ))
    ])
    
    model_pipeline.fit(X, y)
    print(" -> Model successfully trained with optimized hyperparameters.")
    
    # ---------------------------------------------------------
    # 5. Export Artifacts
    # ---------------------------------------------------------
    joblib.dump(model_pipeline, f'{MODEL_SAVE_PATH}/final_model.pkl')
    joblib.dump(team_map, f'{MODEL_SAVE_PATH}/team_map.pkl')
    joblib.dump(final_elo, f'{MODEL_SAVE_PATH}/elo_dict.pkl') 
    joblib.dump(final_h2h, f'{MODEL_SAVE_PATH}/h2h_dict.pkl') 
    
    print("\nSUCCESS: All production artifacts saved to the models/ directory.")

if __name__ == "__main__":
    train_final_model()