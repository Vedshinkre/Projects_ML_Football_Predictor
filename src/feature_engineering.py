"""
Feature Engineering Pipeline.

This script transforms raw football match data into advanced predictive features.
It calculates dynamic Elo ratings (historical strength), rolling averages (short-term form), 
Head-to-Head win rates (rivalries), and integrates external FIFA/Market data.

Crucially, all temporal features are calculated strictly using data available *prior* to 
kick-off to prevent data leakage during model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------
# HELPER: CALCULATE ELO RATINGS (Long-Term Strength)
# ---------------------------------------------------------
def calculate_elo(df):
    """
    Calculates dynamic Elo ratings for all teams. Rating updates after each match 
    based on the result and the relative strength of the opponent.
    """
    print("   -> Calculating Elo Ratings (Team Strength)...")
    
    # Initialize all teams with a baseline Elo of 1500
    current_elo = {team: 1500 for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()}
    k_factor = 20 # Controls how volatile the rating changes are after a single match
    
    home_elos, away_elos, elo_probs = [], [], []
    
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        r_home, r_away = current_elo[home], current_elo[away]
        
        # Store ratings strictly before the match is played
        home_elos.append(r_home)
        away_elos.append(r_away)
        
        # Calculate Expected Win Probability using standard Elo formula
        prob_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        elo_probs.append(prob_home)
        
        # Determine actual result multiplier
        if row['FTR'] == 'H': actual = 1
        elif row['FTR'] == 'D': actual = 0.5
        else: actual = 0
            
        # Update ratings for the NEXT time these teams play
        new_r_home = r_home + k_factor * (actual - prob_home)
        new_r_away = r_away + k_factor * ((1 - actual) - (1 - prob_home))
        
        current_elo[home] = new_r_home
        current_elo[away] = new_r_away

    # Append historical calculations to dataframe
    df['Home_Elo'] = home_elos
    df['Away_Elo'] = away_elos
    df['Elo_Diff'] = np.array(home_elos) - np.array(away_elos)
    df['Elo_Prob_Home'] = elo_probs 
    
    return df

# ---------------------------------------------------------
# HELPER: CALCULATE HEAD-TO-HEAD (rivalry)
# ---------------------------------------------------------
def calculate_h2h(df):
    """
    Tracks the historical win rate of Team A vs Team B.
    Defaults to 50% if the teams have no prior matchups in the dataset.
    """
    print("   -> Calculating Head-to-Head...")
    h2h_history = {}
    h2h_win_rates = []
    
    for idx, row in df.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        pair_key = (home, away)
        
        # Retrieve historical record prior to current match
        history = h2h_history.get(pair_key, [])
        win_rate = sum(history) / len(history) if len(history) > 0 else 0.5
        h2h_win_rates.append(win_rate)
        
        if row['FTR'] == 'H': res_home, res_away = 1, 0
        elif row['FTR'] == 'D': res_home, res_away = 0.5, 0.5
        else: res_home, res_away = 0, 1
            
        # Update dictionaries for both perspectives (Home vs Away, Away vs Home)
        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append(res_home)
        
        rev_key = (away, home)
        if rev_key not in h2h_history: h2h_history[rev_key] = []
        h2h_history[rev_key].append(res_away)
        
    df['My_H2H_Win_Rate'] = h2h_win_rates
    return df

# ---------------------------------------------------------
# HELPER: ROLLING STATS (Short-Term Form)
# ---------------------------------------------------------
def add_rolling_features(df):
    """
    Calculates 3-game rolling averages for goals, shots, and aggression metrics.
    Transforms data from match-level to team-level to calculate form properly.
    """
    print("   -> Calculating Rolling Stats...")
    
    # Split matches into Home and Away perspectives
    home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']].copy()
    away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AS', 'HS', 'AST', 'HST', 'AC', 'HC', 'AF', 'HF', 'AY', 'HY', 'AR', 'HR']].copy()
    
    # Standardize column names (e.g., FTHG -> Goals For (GF) from home perspective)
    col_map = {'HomeTeam': 'Team', 'AwayTeam': 'Team', 'FTHG': 'GF', 'FTAG': 'GA', 'HS': 'Sh', 'AS': 'Opp_Sh', 'HST': 'SoT', 'AST': 'Opp_SoT', 'HC': 'Corn', 'AC': 'Opp_Corn', 'HF': 'Fouls', 'AF': 'Opp_Fouls', 'HY': 'Yellow', 'AY': 'Opp_Yellow', 'HR': 'Red', 'AR': 'Opp_Red'}
    away_map = {'AwayTeam': 'Team', 'FTAG': 'GF', 'FTHG': 'GA', 'AS': 'Sh', 'HS': 'Opp_Sh', 'AST': 'SoT', 'HST': 'Opp_SoT', 'AC': 'Corn', 'HC': 'Opp_Corn', 'AF': 'Fouls', 'HF': 'Opp_Fouls', 'AY': 'Yellow', 'HY': 'Opp_Yellow', 'AR': 'Red', 'HR': 'Opp_Red'}
    
    home_df = home_df.rename(columns=col_map)
    away_df = away_df.rename(columns=away_map)
    
    # Recombine and sort chronologically per team
    team_stats = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # Calculate weighted aggression (Red cards hurt more than Yellows)
    team_stats['Cards'] = team_stats['Yellow'] + (team_stats['Red'] * 2)
    team_stats['Opp_Cards'] = team_stats['Opp_Yellow'] + (team_stats['Opp_Red'] * 2)
    
    stats_cols = ['GF', 'GA', 'Sh', 'Opp_Sh', 'SoT', 'Opp_SoT', 'Corn', 'Opp_Corn', 'Fouls', 'Opp_Fouls', 'Cards', 'Opp_Cards']
    roll_cols = [f"{c}_rolling" for c in stats_cols]
    
    def rolling_averages(group, cols, new_cols):
        # IMPORTANT: closed='left' prevents data leakage by excluding the current match from the rolling average
        rolling = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling
        return group

    team_stats = team_stats.groupby('Team').apply(lambda x: rolling_averages(x, stats_cols, roll_cols))
    team_stats = team_stats.reset_index(drop=True)
    
    # Merge engineered rolling stats back into the main match dataframe
    cols_to_keep = ['Date', 'Team'] + roll_cols
    stats_map = team_stats[cols_to_keep]
    
    df = df.merge(stats_map, left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    df = df.rename(columns={c: f"Home_{c}" for c in roll_cols}).drop(columns=['Team'])
    
    df = df.merge(stats_map, left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    df = df.rename(columns={c: f"Away_{c}" for c in roll_cols}).drop(columns=['Team'])

    # Final rename to match the configuration file expectations
    rename_final = {
        'Home_GF_rolling': 'GF_rolling', 'Home_GA_rolling': 'GA_rolling',
        'Home_Sh_rolling': 'Sh_rolling', 'Home_SoT_rolling': 'SoT_rolling', 
        'Home_Corn_rolling': 'Corn_rolling', 'Home_Cards_rolling': 'Cards_rolling',
        'Home_Fouls_rolling': 'Fouls_rolling',
        'Away_GF_rolling': 'Opp_GF_rolling', 'Away_GA_rolling': 'Opp_GA_rolling',
        'Away_Sh_rolling': 'Opp_Sh_rolling', 'Away_SoT_rolling': 'Opp_SoT_rolling',
        'Away_Corn_rolling': 'Opp_Corn_rolling', 'Away_Cards_rolling': 'Opp_Cards_rolling',
        'Away_Fouls_rolling': 'Opp_Fouls_rolling'
    }
    df = df.rename(columns=rename_final)
    return df

# ---------------------------------------------------------
# HELPER: MERGE FIFA RATINGS (Roster Quality)
# ---------------------------------------------------------
def merge_fifa_ratings(df):
    """
    Joins external FIFA ratings. Calculates gaps in Attack, Defense, Midfield, 
    and applies a Log transformation to Financial gaps.
    """
    print("   -> Merging FIFA Data...")
    try:
        fifa_df = pd.read_csv('data/raw/fifa_ratings_cleaned.csv')
        
        # Create a shared Season_Key. If month > 7 (Aug), it belongs to the *next* year's FIFA release.
        df['Season_Key'] = np.where(df['Date'].dt.month > 7, df['Date'].dt.year + 1, df['Date'].dt.year)
        
        home_fifa = fifa_df.rename(columns={'Team': 'HomeTeam', 'overall': 'Home_Ov', 'attack': 'Home_Att', 'midfield': 'Home_Mid', 'defence': 'Home_Def', 'club_worth_eur': 'Home_Worth', 'starting_xi_average_age': 'Home_Age'})
        df = df.merge(home_fifa, on=['Season_Key', 'HomeTeam'], how='left')
        
        away_fifa = fifa_df.rename(columns={'Team': 'AwayTeam', 'overall': 'Away_Ov', 'attack': 'Away_Att', 'midfield': 'Away_Mid', 'defence': 'Away_Def', 'club_worth_eur': 'Away_Worth', 'starting_xi_average_age': 'Away_Age'})
        df = df.merge(away_fifa, on=['Season_Key', 'AwayTeam'], how='left')
        
        # Fill missing values with league average for safety
        fill_cols = ['Home_Ov', 'Home_Att', 'Home_Mid', 'Home_Def', 'Home_Worth', 'Home_Age', 'Away_Ov', 'Away_Att', 'Away_Mid', 'Away_Def', 'Away_Worth', 'Away_Age']
        for c in fill_cols:
            if c in df.columns: df[c] = df[c].fillna(df[c].mean())

        # Calculate Differentials
        if 'Home_Ov' in df.columns:
            df['Ov_Diff'] = df['Home_Ov'] - df['Away_Ov']
            df['Att_Diff'] = df['Home_Att'] - df['Away_Att']
            df['Mid_Diff'] = df['Home_Mid'] - df['Away_Mid']
            df['Def_Diff'] = df['Home_Def'] - df['Away_Def']
            
            # Use log1p (log(x+1)) for financial data to handle right-skewed distributions 
            # and prevent massive outliers from dominating the model's weights.
            df['Worth_Diff'] = np.log1p(df['Home_Worth']) - np.log1p(df['Away_Worth'])
            df['Age_Diff'] = df['Home_Age'] - df['Away_Age']
            
    except Exception as e:
        print(f"Warning: FIFA Merge skipped due to error: {e}")
        for c in ['Ov_Diff', 'Att_Diff', 'Mid_Diff', 'Def_Diff', 'Worth_Diff', 'Age_Diff']: df[c] = 0
    return df

# ---------------------------------------------------------
# NEW: CONTEXT (League Table Proxy)
# ---------------------------------------------------------
def add_context_features(df):
    """
    Calculates current-season Points Per Game (PPG) to act as a proxy for the live league table.
    """
    print("   -> Calculating Context (Season Consistency)...")
    home_df = df[['Date', 'Season_Key', 'HomeTeam', 'FTR']].rename(columns={'HomeTeam': 'Team'})
    home_df['Points'] = home_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    
    away_df = df[['Date', 'Season_Key', 'AwayTeam', 'FTR']].rename(columns={'AwayTeam': 'Team'})
    away_df['Points'] = away_df['FTR'].map({'H': 0, 'D': 1, 'A': 3})
    
    team_stats = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # IMPORTANT: shift(1) ensures we only average points accumulated *before* the current game
    team_stats['Season_Points'] = team_stats.groupby(['Team', 'Season_Key'])['Points'].transform(lambda x: x.expanding().mean().shift(1))
    
    # First game of the season has no prior points, fill with historical average PPG (approx 1.37)
    team_stats['Season_Points'] = team_stats['Season_Points'].fillna(1.37)

    cols_to_merge = ['Date', 'Team', 'Season_Points']
    df = df.merge(team_stats[cols_to_merge], left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Season_Points': 'Home_PPG'}).drop(columns=['Team'])
    df = df.merge(team_stats[cols_to_merge], left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left').rename(columns={'Season_Points': 'Away_PPG'}).drop(columns=['Team'])
    
    df['PPG_Diff'] = df['Home_PPG'] - df['Away_PPG']    
    return df



# ---------------------------------------------------------
# MAIN PIPELINE ORCHESTRATION
# ---------------------------------------------------------
def prepare_data(raw_df):
    """
    Master function to clean the raw data and sequence the feature engineering steps.
    """
    df = raw_df.copy()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Set the binary target variable (1 = Home Win, 0 = Draw/Away Win)
    df['Target'] = (df['FTR'] == 'H').astype(int)
    
    # 1. Base encodings
    df['Venue_code'] = 1
    df['Day_code'] = df['Date'].dt.dayofweek
    df['Hour'] = df['Time'].apply(lambda x: int(x.split(':')[0]) if isinstance(x, str) else 15)
    
    # 2. Add engineered features
    df = calculate_elo(df)
    df = calculate_h2h(df)
    df = add_rolling_features(df)
    df = merge_fifa_ratings(df)
    df = add_context_features(df)
    
    
    # 3. Create Expected Goal Differential (xGD) Proxy based on Shots on Target
    df['xGD_Proxy'] = (df['SoT_rolling']/4.0) - (df['Opp_SoT_rolling']/4.0)
    
    # Fill any remaining NaNs (primarily first 3 games of the dataset missing rolling averages)
    df = df.fillna(0)
    
    # 4. Generate Opponent Code mappings securely
    df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
    unique_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    all_teams = [t for t in unique_teams if isinstance(t, str)]
    all_teams.sort()
    team_map = {team: i for i, team in enumerate(all_teams)}
    df['Opp_code'] = df['AwayTeam'].map(team_map).fillna(0).astype(int)
    
    return df, team_map

if __name__ == "__main__":
    # Self-test block: Processes raw data and saves a feature-rich output for model training
    try:
        raw = pd.read_csv('data/raw/premierleague_10yrs.csv')
        processed, _ = prepare_data(raw)
        processed.to_csv('data/processed/full_data_with_features.csv', index=False)
        print("Success: Saved engineered dataset to data/processed/")
    except Exception as e:
        print(f"Test failed: {e}")