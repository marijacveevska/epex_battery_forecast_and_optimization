import os
import time
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = EntsoePandasClient(api_key=os.getenv('ENTSO_E_API_KEY'))

# Settings
NL    = '10YNL----------L'
START = pd.Timestamp('2021-01-01', tz='Europe/Amsterdam')
END   = pd.Timestamp('2026-01-01', tz='Europe/Amsterdam')
OUT   = 'Data/raw/entso_e'

def fetch_and_save(name, func, *args, **kwargs):
    print(f"Fetching {name}...")
    try:
        data = func(*args, **kwargs)
        path = f"{OUT}/{name}.csv"
        data.to_csv(path)
        print(f"  Done — {len(data)} rows saved to {path}")
        time.sleep(3)
        return data
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

# 1. Actual total load
fetch_and_save(
    'nl_actual_load',
    client.query_load,
    NL, start=START, end=END
)

# 2. Day-ahead load forecast
fetch_and_save(
    'nl_load_forecast',
    client.query_load_forecast,
    NL, start=START, end=END
)

print("\nAll done. Check Data/raw/entso_e/ for your files.")

# 3. Actual generation per production type

years = [2021, 2022, 2023, 2024, 2025]
all_years = []

for year in years:
    START = pd.Timestamp(f'{year}-01-01', tz='Europe/Amsterdam')
    END   = pd.Timestamp(f'{year+1}-01-01', tz='Europe/Amsterdam')
    
    print(f"Fetching generation mix for {year}...")
    try:
        data = client.query_generation(NL, start=START, end=END)
        all_years.append(data)
        print(f"  Done — {len(data)} rows")
        time.sleep(5)
    except Exception as e:
        print(f"  ERROR for {year}: {e}")
        time.sleep(10)

if all_years:
    combined = pd.concat(all_years)
    path = f"{OUT}/nl_generation_mix.csv"
    combined.to_csv(path)
    print(f"\nSaved {len(combined)} total rows to {path}")
else:
    print("Nothing fetched successfully")

print("\nDone.")