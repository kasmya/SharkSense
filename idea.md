#!/bin/bash
# ============================================================================
# SharkSense - Complete End‑to‑End Pipeline
# Predicts shark attack fatality risk using environmental data.
# Run this script in an empty directory where you have placed:
#   geocoded-global-shark-attacks.csv   (from Kaggle)
# Outputs: enriched_attacks.csv, model.pkl, evaluation report.
# ============================================================================

set -e  # exit on error

echo "============================================"
echo " SharkSense – Global Shark Attack Risk Model"
echo "============================================"

# 1. Create and activate virtual environment
echo "[1/6] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
echo "[2/6] Installing required Python packages..."
pip install --upgrade pip
pip install pandas numpy scikit-learn openmeteo-requests retry-requests
pip install geopandas shapely folium streamlit streamlit-folium matplotlib seaborn

# 3. Download coastline data (Natural Earth 110m)
echo "[3/6] Downloading coastline shapefile..."
wget -q -N https://naturalearth.s3.amazonaws.com/110m_physical/ne_110m_coastline.zip
unzip -o -q ne_110m_coastline.zip -d coastline_data/

# 4. Create the Python processing script
echo "[4/6] Creating SharkSense Python pipeline (sharksense_pipeline.py)..."
cat > sharksense_pipeline.py << 'EOF'
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# Geospatial imports
import geopandas as gpd
from shapely.geometry import Point

# API and date handling
import openmeteo_requests
import requests
from retry_requests import retry
from datetime import datetime, timedelta, date
import time
import math

# Machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

print("Loading data...")
# ------------------------------- DATA LOADING -------------------------------
try:
    df_raw = pd.read_csv("geocoded-global-shark-attacks.csv")
except FileNotFoundError:
    raise FileNotFoundError("Please place geocoded-global-shark-attacks.csv in the current directory.")

# Keep only needed columns
df = df_raw[['NEW_Latitude_Location_Area_Country', 'NEW_Longitude_Location_Area_Country',
             'Date', 'fatal_y_n']].copy()
df.rename(columns={
    'NEW_Latitude_Location_Area_Country': 'latitude',
    'NEW_Longitude_Location_Area_Country': 'longitude'
}, inplace=True)

# Drop rows with missing coordinates or date
df = df.dropna(subset=['latitude', 'longitude', 'Date', 'fatal_y_n'])
df['incident_date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['incident_date'])
print(f"Initial records: {len(df)}")

# ------------------------------- COORDINATE SNAPPING -------------------------------
print("Snapping inland points to coastline...")
coastline = gpd.read_file("coastline_data/ne_110m_coastline.shp")

def snap_to_coast(lat, lon, max_dist_km=50):
    point = Point(lon, lat)
    # Project to a metric CRS for accurate distance? Simpler: approximate degrees->km
    distances = coastline.distance(point)
    min_idx = distances.idxmin()
    min_dist_deg = distances[min_idx]
    min_dist_km = min_dist_deg * 111.0
    if min_dist_km > max_dist_km:
        return None, None
    nearest_line = coastline.loc[min_idx, 'geometry']
    nearest_point = nearest_line.interpolate(nearest_line.project(point))
    return nearest_point.y, nearest_point.x

snapped_lats, snapped_lons = [], []
for idx, row in df.iterrows():
    new_lat, new_lon = snap_to_coast(row['latitude'], row['longitude'])
    snapped_lats.append(new_lat)
    snapped_lons.append(new_lon)
    if (idx+1) % 500 == 0:
        print(f"  Snapped {idx+1}/{len(df)} rows")
df['snapped_lat'] = snapped_lats
df['snapped_lon'] = snapped_lons
df = df.dropna(subset=['snapped_lat', 'snapped_lon'])
print(f"After snapping (within 50km of coast): {len(df)} records")

# ------------------------------- ENVIRONMENTAL FEATURES -------------------------------
print("Initialising API clients (Open-Meteo)...")
session = retry(requests.Session(), retries=3, backoff_factor=1)
openmeteo = openmeteo_requests.Client(session=session)

def get_moon_illumination(date_obj):
    days_since_new = (date_obj - datetime(2000, 1, 6)).days
    cycle = 29.53
    phase = (days_since_new % cycle) / cycle
    illumination = (1 - math.cos(2 * math.pi * phase)) / 2
    return round(illumination * 100)

def get_marine_weather(lat, lon, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    result = {'sst_c': np.nan, 'air_temp_c': np.nan, 'wind_kmh': np.nan}
    # SST
    marine_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "sea_surface_temperature"
    }
    try:
        marine_resp = openmeteo.weather_api("https://marine-api.open-meteo.com/v1/marine", params=marine_params)
        if marine_resp:
            sst_vals = marine_resp[0].Hourly().Variables(0).ValuesAsNumpy()
            if sst_vals is not None and not np.isnan(sst_vals).all():
                result['sst_c'] = float(np.nanmean(sst_vals))
    except:
        pass
    # Air temp + wind
    hist_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ["temperature_2m", "wind_speed_10m"]
    }
    try:
        hist_resp = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params=hist_params)
        if hist_resp:
            hourly = hist_resp[0].Hourly()
            temp_vals = hourly.Variables(0).ValuesAsNumpy()
            wind_vals = hourly.Variables(1).ValuesAsNumpy()
            if temp_vals is not None and not np.isnan(temp_vals).all():
                result['air_temp_c'] = float(np.nanmean(temp_vals))
            if wind_vals is not None and not np.isnan(wind_vals).all():
                result['wind_kmh'] = float(np.nanmean(wind_vals))
    except:
        pass
    if np.isnan(result['sst_c']) and np.isnan(result['air_temp_c']) and np.isnan(result['wind_kmh']):
        return None
    return result

# Cache to avoid repeated API calls
cache_file = "weather_cache.csv"
if os.path.exists(cache_file):
    cache_df = pd.read_csv(cache_file)
else:
    cache_df = pd.DataFrame(columns=['lat','lon','date','sst_c','air_temp_c','wind_kmh'])

def get_cached_or_fetch(lat, lon, target_date):
    date_str = target_date.strftime('%Y-%m-%d')
    match = cache_df[(cache_df['lat'] == lat) & (cache_df['lon'] == lon) & (cache_df['date'] == date_str)]
    if not match.empty:
        return {'sst_c': match.iloc[0]['sst_c'], 'air_temp_c': match.iloc[0]['air_temp_c'], 'wind_kmh': match.iloc[0]['wind_kmh']}
    weather = get_marine_weather(lat, lon, target_date)
    if weather:
        new_row = pd.DataFrame([{'lat': lat, 'lon': lon, 'date': date_str,
                                 'sst_c': weather['sst_c'], 'air_temp_c': weather['air_temp_c'],
                                 'wind_kmh': weather['wind_kmh']}])
        global cache_df
        cache_df = pd.concat([cache_df, new_row], ignore_index=True)
        cache_df.to_csv(cache_file, index=False)
    return weather

# Enrich rows (limit to first 1000 rows for demo, remove limit for full run)
print("Fetching environmental data (this may take several minutes)...")
df['sst_c'] = np.nan
df['air_temp_c'] = np.nan
df['wind_kmh'] = np.nan
df['moon_pct'] = np.nan

total = len(df)
for i, (idx, row) in enumerate(df.iterrows()):
    moon = get_moon_illumination(row['incident_date'])
    df.at[idx, 'moon_pct'] = moon
    
    weather = get_cached_or_fetch(row['snapped_lat'], row['snapped_lon'], row['incident_date'])
    if weather:
        df.at[idx, 'sst_c'] = weather['sst_c']
        df.at[idx, 'air_temp_c'] = weather['air_temp_c']
        df.at[idx, 'wind_kmh'] = weather['wind_kmh']
    
    if (i+1) % 50 == 0 or (i+1) == total:
        print(f"  Processed {i+1}/{total} rows")
    time.sleep(0.2)   # gentle rate limiting

# Save enriched dataset
df.to_csv("enriched_attacks.csv", index=False)
print("Saved enriched_attacks.csv")

# ------------------------------- MACHINE LEARNING -------------------------------
print("Training Random Forest model...")
model_df = df.dropna(subset=['sst_c', 'air_temp_c', 'wind_kmh', 'moon_pct']).copy()
model_df['fatal'] = (model_df['fatal_y_n'].astype(str).str.upper() == 'Y').astype(int)

features = ['sst_c', 'air_temp_c', 'wind_kmh', 'moon_pct']
X = model_df[features]
y = model_df['fatal']

if len(model_df) < 10:
    print("WARNING: Not enough data for meaningful training. Exiting.")
    exit(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-fatal', 'Fatal']))
print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f}")

importances = model.feature_importances_
print("\nFeature importances:")
for f, imp in zip(features, importances):
    print(f"  {f}: {imp:.3f}")

# Save model
joblib.dump(model, "sharksense_model.pkl")
print("Model saved as sharksense_model.pkl")

print("\n✅ Pipeline completed successfully!")
EOF

# 5. Run the Python pipeline
echo "[5/6] Running SharkSense pipeline (this may take 10-20 minutes)..."
python sharksense_pipeline.py

# 6. (Optional) Launch dashboard – commented out as it requires manual interaction
# echo "[6/6] To launch the interactive dashboard, run: streamlit run dashboard.py"
# echo " (First create dashboard.py using the code from the project report)"

echo "============================================"
echo " All done! Output files:"
echo "   - enriched_attacks.csv"
echo "   - sharksense_model.pkl"
echo "   - weather_cache.csv"
echo "============================================"
