"""
Configuration File for Feature Selection and Project Paths.

The config file adds modularity and substitutes all the parameters separately,
 so the model doesn't need to be hardcoded and is open for further exploration and expansion. 

The levels represent different approaches that I came up with as possible parameters for building
 the machine learning model:
"""

# ---------------------------------------------------------
# FEATURE SET DICTIONARY
# ---------------------------------------------------------
FEATURE_SETS = {
    
    # Baseline Model: Uses only static temporal and categorical data.
    "Level 1 (Static)": [
        "Venue_code", "Opp_code", "Hour", "Day_code"
    ],
    
    # Short-Term Form: Adds basic rolling averages for the team's offensive/defensive output.
    "Level 2 (Rolling)": [
        "Venue_code", "Opp_code", "Hour", "Day_code", 
        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling"
    ],
    
    # Extended Form: Includes aggression metrics (Corners, Fouls, Cards).
    "Level 3 (Physical)": [
        "Venue_code", "Opp_code", "Hour", "Day_code", 
        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling",
        "Corn_rolling", "Fouls_rolling", "Cards_rolling"
    ],

    # Matchup Dynamics: Introduces the opponent's current form to calculate relative strength.
    "Level 4 (Pure Matchup)": [
        "Venue_code", "Opp_code", "Hour", "Day_code", 
        
        # Team's own form
        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling",
        "Corn_rolling", "Fouls_rolling", "Cards_rolling",
        
        # Opponent's form
        "Opp_GF_rolling", "Opp_GA_rolling", "Opp_Sh_rolling", "Opp_SoT_rolling",
        "Opp_Corn_rolling", "Opp_Fouls_rolling", "Opp_Cards_rolling"
    ],
    
    # Advanced Analytics: Integrates long-term strength (Elo), rivalry (H2H), and Expected Goal differentials.
    "Level 5 (Professional)": [  
        "Venue_code", "Opp_code", "Hour", "Day_code",
        "Elo_Diff", "My_H2H_Win_Rate", "xGD_Proxy",
        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling", "Corn_rolling", "Cards_rolling",
        "Opp_GF_rolling", "Opp_GA_rolling", "Opp_Sh_rolling", "Opp_SoT_rolling", "Opp_Corn_rolling", "Opp_Cards_rolling"
    ],
    
    # Roster Quality: Brings in external EA Sports FIFA ratings of the teams to measure actual squad talent and financial gaps.
    "Level 6 (Ultimate)": [
        "Venue_code", "Opp_code", "Hour", "Day_code",
        "Elo_Diff", "My_H2H_Win_Rate", "xGD_Proxy",
        
        # External Squad Quality Metrics (FIFA)
        "Ov_Diff", "Att_Diff", "Mid_Diff", "Def_Diff", "Worth_Diff", "Age_Diff",

        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling", "Corn_rolling",
        "Opp_GF_rolling", "Opp_GA_rolling", "Opp_Sh_rolling", "Opp_SoT_rolling"
    ],
    
    # Seasonal Context: Adds League Table positioning proxy (Points Per Game Differential).
    "Level 7 (Context)": [
        "Venue_code", "Opp_code",
        "Elo_Diff", "Elo_Prob_Home",
        "My_H2H_Win_Rate",
        
        # Current Season standing
        "PPG_Diff",       
        
        # Squad Class
        "Ov_Diff", "Att_Diff", "Mid_Diff", "Def_Diff", "Worth_Diff", "Age_Diff",

        # Form / Aggression
        "GF_rolling", "GA_rolling", "Sh_rolling", "SoT_rolling", "Corn_rolling",
        "Opp_GF_rolling", "Opp_GA_rolling", "Opp_Sh_rolling", "Opp_SoT_rolling",
        "Fouls_rolling", "Cards_rolling", "Opp_Fouls_rolling", "Opp_Cards_rolling"
    ]
}

# ---------------------------------------------------------
# PROJECT FILE PATHS
# ---------------------------------------------------------
# Path to the historical dataset used for training
TRAIN_DATA_PATH = 'data/raw/premierleague_10yrs.csv'

# Directory where trained models (.pkl) will be saved
MODEL_SAVE_PATH = 'models'

# Endpoint for fetching live/current season data for predictions
DATA_URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"