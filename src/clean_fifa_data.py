"""
FIFA Data Processing & Imputation Pipeline.

This script acts as an ETL (Extract, Transform, Load) pipeline for raw EA Sports FIFA data.
It standardizes team names to match our Premier League dataset, resolves integer overflow 
issues in financial columns, filters for relevant leagues, and applies a smart imputation 
strategy to estimate missing club valuations based on their overall squad ratings.
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------
# CONFIGURATION & MAPPING
# ---------------------------------------------------------
FIFA_FILE_PATH = 'data/raw/male_teams.csv'
PL_DATA_PATH = 'data/raw/premierleague_10yrs.csv'
OUTPUT_PATH = 'data/raw/fifa_ratings_cleaned.csv'

# Core columns required for the ML model's "Ultimate" level features
COLS_TO_KEEP = [
    'team_name', 'fifa_version', 'fifa_update_date', 
    'overall', 'attack', 'midfield', 'defence', 
    'club_worth_eur', 'starting_xi_average_age', 'league_name'
]

# Entity Resolution: Mapping EA Sports naming conventions to football-data.co.uk conventions
NAME_MAPPING = {
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Brighton & Hove Albion": "Brighton",
    "Leicester City": "Leicester",
    "Norwich City": "Norwich",
    "Leeds United": "Leeds",
    "Nottingham Forest": "Nott'm Forest",
    "Luton Town": "Luton",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Swansea City": "Swansea",
    "Stoke City": "Stoke",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Huddersfield Town": "Huddersfield",
    "Sheffield United": "Sheffield United",
    "Blackburn Rovers": "Blackburn",
    "Wigan Athletic": "Wigan",
    "Bolton Wanderers": "Bolton",
    "Birmingham City": "Birmingham",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Burnley": "Burnley",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich",
    "Middlesbrough": "Middlesbrough",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Watford": "Watford"
}

# Used to filter out lower divisions and foreign leagues to save memory and processing time
VALID_LEAGUES = [
    "Premier League", "Barclays Premier League", "English Premier League",
    "EFL Championship", "Championship", "Football League Championship"
]

def clean_fifa_data():
    """
    Executes the cleaning pipeline: loads data, handles data types, scales financials, 
    maps entities, imputes missing values, and saves the cleaned dataset.
    """
    print("Starting FIFA Data Cleaning (Scaling financials to Millions)...")
    
    # ---------------------------------------------------------
    # 1. DATA INGESTION
    # ---------------------------------------------------------
    try:
        # low_memory=False prevents pandas from guessing dtypes in mixed-type columns
        fifa_df = pd.read_csv(FIFA_FILE_PATH, low_memory=False)
        pl_df = pd.read_csv(PL_DATA_PATH)
        print(f"Loaded {len(fifa_df)} FIFA rows.")
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files. {e}")
        return

    # ---------------------------------------------------------
    # 2. TYPE CORRECTION & SCALING (Handling Integer Overflows)
    # ---------------------------------------------------------
    # The raw FIFA dataset often suffers from 32-bit integer overflows for highly valued 
    # clubs, resulting in negative club values. We must reset these to NaN and scale.
    cols_to_check = ['club_worth_eur', 'overall', 'attack', 'midfield', 'defence']
    for col in cols_to_check:
        if col in fifa_df.columns:
            # Force conversion to numeric, turning bad string data into NaN
            fifa_df[col] = pd.to_numeric(fifa_df[col], errors='coerce')
            
            if col == 'club_worth_eur':
                # Nullify invalid negative values caused by integer overflow
                fifa_df.loc[fifa_df[col] <= 0, col] = np.nan
                
                # Scale from Euros to Millions of Euros to help the ML model process weights
                # Example: 949,000,000 -> 949.0
                fifa_df[col] = fifa_df[col] / 1_000_000.0

    # ---------------------------------------------------------
    # 3. LEAGUE & TEAM FILTERING
    # ---------------------------------------------------------
    # Filter strictly for English leagues
    if 'league_name' in fifa_df.columns:
        pattern = '|'.join(VALID_LEAGUES)
        fifa_df = fifa_df[fifa_df['league_name'].str.contains(pattern, case=False, na=False)]

    # Standardize team names and drop any team that hasn't played in the PL dataset
    fifa_df['team_name_cleaned'] = fifa_df['team_name'].replace(NAME_MAPPING)
    real_teams = set(pl_df['HomeTeam'].unique()) | set(pl_df['AwayTeam'].unique())
    fifa_df = fifa_df[fifa_df['team_name_cleaned'].isin(real_teams)]
    
    # ---------------------------------------------------------
    # 4. TEMPORAL AGGREGATION
    # ---------------------------------------------------------
    # Teams get multiple updates per FIFA version. We want the final rating of that season.
    if 'fifa_update_date' in fifa_df.columns:
        fifa_df = fifa_df.sort_values('fifa_update_date')
    
    # Group by game version and team, keeping only the most recent update row
    fifa_unique = fifa_df.groupby(['fifa_version', 'team_name_cleaned'], as_index=False).last()
    
    # ---------------------------------------------------------
    # 5. SMART IMPUTATION
    # ---------------------------------------------------------
    def estimate_worth_millions(row):
        """
        Fallback logic for missing financial data. If a club's worth was lost to 
        integer overflow, we estimate its value tier based on its overall squad rating.
        """
        # If the data is valid and positive, keep it
        if pd.notnull(row['club_worth_eur']) and row['club_worth_eur'] > 0:
            return row['club_worth_eur']
        
        # Determine fallback value (in millions) based on squad rating tiers
        rating = row['overall']
        if pd.isnull(rating): return 150.0  # Safe median for complete missing data
        
        if rating >= 85: return 1500.0  # Elite tier (e.g., Man City, Arsenal)
        if rating >= 83: return 1000.0  # Champions League tier
        if rating >= 80: return 600.0   # Upper Mid-table tier
        if rating >= 76: return 300.0   # Mid-table tier
        return 150.0                    # Lower PL / Championship tier

    print("Applying 'Smart Worth Estimation' to impute missing financial data...")
    fifa_unique['club_worth_eur'] = fifa_unique.apply(estimate_worth_millions, axis=1)

    # Impute any remaining missing categorical stats with the column median
    for col in ['overall', 'attack', 'midfield', 'defence']:
        fifa_unique[col] = fifa_unique[col].fillna(fifa_unique[col].median())

    # ---------------------------------------------------------
    # 6. FEATURE ENGINEERING & EXPORT
    # ---------------------------------------------------------
    # Create a uniform 'Season_Key' (e.g., FIFA 23 -> 2023) to join with match data
    if 'fifa_version' in fifa_unique.columns:
        fifa_unique['Season_Key'] = 2000 + fifa_unique['fifa_version']
    
    # Finalize columns for export
    fifa_unique = fifa_unique.rename(columns={'team_name_cleaned': 'Team'})
    cols_to_save = ['Season_Key', 'Team', 'overall', 'attack', 'midfield', 'defence', 
                    'club_worth_eur', 'starting_xi_average_age']
    
    final_cols = [c for c in cols_to_save if c in fifa_unique.columns]
    fifa_unique[final_cols].to_csv(OUTPUT_PATH, index=False)
    
    print(f"Success! Cleaned FIFA data saved to: {OUTPUT_PATH}")
    
    # ---------------------------------------------------------
    # VERIFICATION OUTPUT
    # ---------------------------------------------------------
    print("Verification Check: Ensuring values scaled to millions correctly...")
    check = fifa_unique[
        (fifa_unique['Team'].isin(['Man City', 'Liverpool', 'Luton'])) & 
        (fifa_unique['Season_Key'] == 2023)
    ]
    if not check.empty:
        print(check[['Team', 'Season_Key', 'overall', 'club_worth_eur']].to_string(index=False))

if __name__ == "__main__":
    clean_fifa_data()