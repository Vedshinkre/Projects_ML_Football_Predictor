import pandas as pd
import requests
import io
import os

# ---------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ---------------------------------------------------------
# Define the seasons to download. 
# The format '1415' corresponds to the 2014/2015 season.
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"]

# Base URL for football-data.co.uk. 'E0' denotes the English Premier League.
BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

# Destination path for the aggregated dataset
OUTPUT_FILE = "data/raw/premierleague_10yrs.csv"

def download_historical_data():
    """
    Iterates through the specified seasons, downloads the CSV data directly 
    into memory, standardizes it, and compiles it into a single master dataset.
    """
    all_data = []
    
    print(f"Starting download for {len(SEASONS)} seasons...")
    
    for season in SEASONS:
        # Inject the season string into the URL placeholder
        url = BASE_URL.format(season)
        print(f"  Downloading: {url} ...", end=" ")
        
        try:
            # Execute the GET request to fetch the file
            response = requests.get(url)
            
            # Check if the request was successful (HTTP 200 OK)
            if response.status_code == 200:
                
                # Convert the raw bytes into a pandas DataFrame.
                # We use 'iso-8859-1' encoding because older football-data files 
                # contain special European characters that can crash standard UTF-8 decoders.
                # io.StringIO allows pandas to read the string exactly like a physical CSV file in memory.
                df = pd.read_csv(io.StringIO(response.content.decode('iso-8859-1')))
                
                # Track the season ID to ensure we can filter or group by season later
                df['SeasonId'] = season
                
                # Store the DataFrame in our master list for later concatenation
                all_data.append(df)
                print("Done")
            else:
                # Catch 404s or server errors without crashing the entire loop
                print(f"Failed (Status {response.status_code})")
                
        except Exception as e:
            # Catch connection errors (e.g., timeout, no internet connection)
            print(f"Error: {e}")

    # ---------------------------------------------------------
    # DATA AGGREGATION & EXPORT
    # ---------------------------------------------------------
    # Only proceed if we actually downloaded at least one season successfully
    if all_data:
        # Combine all individual season DataFrames into one large DataFrame.
        # ignore_index=True resets the row numbers from 0 to the length of the new dataframe.
        full_df = pd.concat(all_data, ignore_index=True)
        
        # Ensure the output directory structure exists (creates 'data/raw/' if missing)
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        
        # Save the combined dataset to disk without the pandas index column
        full_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nSUCCESS! Saved {len(full_df)} games to {OUTPUT_FILE}")
        print(f"  Columns available: {list(full_df.columns[:10])}...")
    else:
        print("\nNo data downloaded.")

if __name__ == "__main__":
    download_historical_data()