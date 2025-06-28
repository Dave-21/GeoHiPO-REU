import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import LineString, Point
from collections import Counter
from sklearn.cluster import KMeans
from math import radians, cos, sin, asin, sqrt
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ------------------------
# Parameters
# ------------------------
ACCURACY_THRESHOLD_KM = 500  # Used for "correct" predictions

# ------------------------
# Load and Prepare Data
# ------------------------
PREDICTIONS_FILE = "city_country_predictions.json"
GROUND_TRUTH_FILE = "im2gps3k.csv"

if PREDICTIONS_FILE.endswith(".json"):
    with open(PREDICTIONS_FILE, "r") as f:
        preds = json.load(f)
    pred_df = pd.DataFrame(preds)
else:
    pred_df = pd.read_csv(PREDICTIONS_FILE)
    if "prediction" in pred_df.columns:
        pred_df["prediction"] = pred_df["prediction"].apply(
            lambda x: json.loads(x)[0] if isinstance(x, str) and x.strip().startswith("[") else x
        )

pred_df["guessed_place"] = pred_df["prediction"].apply(lambda x: x[0] if isinstance(x, list) else x)

gt_df = pd.read_csv(GROUND_TRUTH_FILE, usecols=["IMG_ID", "LAT", "LON"])

df = pred_df.merge(gt_df, left_on="filename", right_on="IMG_ID", how="left")

geolocator = Nominatim(user_agent="geo_visualizer")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

records = []

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 6371 * 2 * asin(sqrt(a))

cache = {}

def geocode_place(place):
    if not isinstance(place, str) or not place.strip():
        return None, None
    if place in cache:
        return cache[place]
    try:
        loc = geocode(place)
        if loc:
            cache[place] = (loc.latitude, loc.longitude)
            return cache[place]
    except Exception as e:
        print(f"Geocoding error for '{place}': {e}")
    cache[place] = (None, None)
    return cache[place]

for _, row in df.iterrows():
    has_valid_gt = not pd.isna(row["LAT"]) and not pd.isna(row["LON"])
    if not has_valid_gt:
        continue

    lat_pred, lon_pred = geocode_place(row.get("guessed_place"))
    has_valid_coords = lat_pred is not None and lon_pred is not None

    if has_valid_coords:
        error_km = haversine(lon_pred, lat_pred, row["LON"], row["LAT"])
    else:
        error_km = float("inf")

    guessed = row.get("guessed_place", "Unknown") or "Unknown"
    is_missing_guess = guessed.strip().lower() in ["", "unknown", "none", "null"]

    records.append({
        "image": row["filename"],
        "guessed_place": guessed,
        "continent": None,
        "lat_pred": lat_pred,
        "lon_pred": lon_pred,
        "lat_true": row["LAT"],
        "lon_true": row["LON"],
        "error_km": error_km,
        "correct": (error_km <= ACCURACY_THRESHOLD_KM) and not is_missing_guess,
    })



df = pd.DataFrame(records)
os.makedirs("output", exist_ok=True)

# ------------------------
# Summary Stats
# ------------------------
summary = [
    f"Total predictions: {len(df)}",
    f"Accuracy threshold: {ACCURACY_THRESHOLD_KM} km",
    f"Correct predictions: {(df['correct'].mean() * 100):.2f}%",
    f"Mean error: {df.error_km.mean():.2f} km",
    f"Median error: {df.error_km.median():.2f} km",
    f"Std deviation: {df.error_km.std():.2f} km",
    ""
]

summary.append("\nPrediction Accuracy by Distance Range:")
for r in [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000]:
    pct = (df["error_km"] <= r).mean() * 100
    summary.append(f"  ≤ {r:>4} km: {pct:5.2f}%")
summary.append("\n")

# Top city accuracy
top_cities = df["guessed_place"].value_counts().head(10).index.tolist()
city_acc = df[df["guessed_place"].isin(top_cities)].groupby("guessed_place")["correct"].mean() * 100

summary.append("Top Guessed Cities (with Accuracy):")
for city in top_cities:
    summary.append(f"  {city}: {city_acc[city]:.2f}% accuracy ({df[df['guessed_place'] == city].shape[0]} times)")
summary.append("")

# Cluster insights
coords = df[["lat_true", "lon_true"]].to_numpy()
kmeans = KMeans(n_clusters=6, random_state=42).fit(coords)
df["cluster"] = kmeans.labels_
cluster_stats = df.groupby("cluster")["error_km"].agg(["mean", "median", "count"])
summary.append("Error Clusters Summary:\n")
summary.append(cluster_stats.to_string())

with open("output/geo_hipo_summary.txt", "w") as f:
    f.write("\n".join(summary))

# ------------------------
# Visualizations
# ------------------------
world = gpd.read_file("data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp")

# 1. World Heatmap (All predictions)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon_pred, df.lat_pred))
fig, ax = plt.subplots(figsize=(14, 10))
world.plot(ax=ax, color='lightgray', edgecolor='black')
gdf.plot(ax=ax, color='red', alpha=0.3, markersize=10, label='Predicted Locations')
plt.title("Prediction Density: All Predictions")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.savefig("output/world_accuracy_heatmap.png")
plt.close()

# 2. World Heatmap (Only Correct)
correct_gdf = gdf[df["correct"]]
fig, ax = plt.subplots(figsize=(14, 10))
world.plot(ax=ax, color='lightgray', edgecolor='black')
correct_gdf.plot(ax=ax, color='green', alpha=0.4, markersize=10, label=f"Correct (≤{ACCURACY_THRESHOLD_KM} km)")
plt.title("Correct Predictions by Geolocation")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.savefig("output/world_correct_predictions.png")
plt.close()

# 3. World Error Lines – per bin
bins = [0, 50, 250, 500, 1000, 5000]
df["error_bin"] = pd.cut(df["error_km"], bins=bins, labels=False)
colors = sns.color_palette("coolwarm", n_colors=len(bins))

for b in range(len(bins) - 1):
    subset = df[df["error_bin"] == b]
    if subset.empty:
        continue
    lines = [LineString([(row.lon_true, row.lat_true), (row.lon_pred, row.lat_pred)]) for _, row in subset.iterrows()]
    gsub = gpd.GeoDataFrame(subset, geometry=lines)

    fig, ax = plt.subplots(figsize=(16, 10))
    world.plot(ax=ax, color="white", edgecolor="gray")
    gsub.plot(ax=ax, color=colors[b], linewidth=3.0, alpha=0.9, label=f"{bins[b]}–{bins[b+1]} km")
    plt.title(f"Error Vectors for Predictions ({bins[b]}–{bins[b+1]} km)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"output/world_error_lines_bin_{b}.png")
    plt.close()

# 4. Error Histogram
plt.figure(figsize=(10, 6))
sns.histplot(df["error_km"], bins=40, kde=True, log_scale=(False, True))
plt.xlabel("Prediction Error (km)")
plt.ylabel("Count (log scale)")
plt.title("Histogram of Prediction Errors")
plt.grid(True)
plt.savefig("output/error_histogram.png")
plt.close()

# 5. Error horizontal boxplot with jittered points
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="error_km", color="skyblue", showfliers=False)
sns.stripplot(data=df[df["error_km"] < 5000], x="error_km", color="black", alpha=0.2, jitter=0.2)
plt.xlabel("Prediction Error (km)")
plt.title("Prediction Error Distribution (Boxplot + Scatter)")
plt.xlim(0, 5000)
plt.tight_layout()
plt.savefig("output/error_boxplot_swarm.png")
plt.close()

# 6. Continent Accuracy
acc_by_cont = df.groupby("continent")["correct"].mean().sort_values() * 100
plt.figure(figsize=(10, 6))
sns.barplot(x=acc_by_cont.index, y=acc_by_cont.values, palette="viridis")
plt.ylabel("Accuracy (%)")
plt.title(f"Continent-wise Accuracy (≤{ACCURACY_THRESHOLD_KM} km)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("output/continent_accuracy_barplot.png")
plt.close()

# 7. City Frequency
top_counts = df["guessed_place"].value_counts().nlargest(10).sort_values()
top_counts.plot(kind="barh", color="teal", figsize=(10, 5))
plt.xlabel("Frequency")
plt.title("Most Guessed Cities")
plt.tight_layout()
plt.savefig("output/city_frequency_barplot.png")
plt.close()

# 8. City Accuracy (new)
top_acc = df[df["guessed_place"].isin(top_cities)].groupby("guessed_place")["correct"].mean().sort_values() * 100
top_acc.plot(kind="barh", color="green", figsize=(10, 5))
plt.xlabel("Accuracy (%)")
plt.title(f"Accuracy of Top Guessed Cities (≤{ACCURACY_THRESHOLD_KM} km)")
plt.tight_layout()
plt.savefig("output/city_accuracy_barplot.png")
plt.close()
