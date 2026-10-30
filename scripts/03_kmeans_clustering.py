"""
Moosic Project — Step 3: K-Means Clustering
============================================
K-Means partitions songs into k groups by minimising the within-cluster
sum of squared distances (inertia).

Pipeline:
  1. Elbow Method  — plot inertia for k = 1 … 19, identify the "elbow"
  2. Silhouette Score — validate cluster quality for k = 2 … 14
  3. Choose k = 6 — best balance of statistical quality and musical variety
  4. Fit final model, inspect cluster sizes and centroids
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH = "../spotify_5000_songs.csv"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]
BEST_K     = 6
RANDOM_STATE = 42

CLUSTER_NAMES = {
    0: "Energetic Rock",
    1: "Instrumental Ambient",
    2: "Acoustic & Mellow",
    3: "Hard & Intense",
    4: "Pop & Dance",
    5: "Hip-Hop & R&B",
}

# ─── 1. Load & Scale ───────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── 2. Elbow Method ───────────────────────────────────────────────────────
print("Running Elbow Method (k = 1 … 19) …")
inertias = []
k_range  = range(1, 20)

for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(list(k_range), inertias, "o-", color="#1DB954", linewidth=2.5, markersize=8)
ax.axvline(BEST_K, color="#FF6B35", linewidth=2, linestyle="--", label=f"Chosen k = {BEST_K}")
ax.fill_between(list(k_range), inertias, alpha=0.08, color="#1DB954")
ax.set_xlabel("Number of Clusters  k", fontsize=13)
ax.set_ylabel("Inertia (WCSS)", fontsize=13)
ax.set_title("Elbow Method — Optimal k Selection", fontsize=15, fontweight="bold")
ax.legend(fontsize=12)
ax.grid(True, alpha=0.4)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "03_elbow_method.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Saved] {out_path}")

# ─── 3. Silhouette Scores ──────────────────────────────────────────────────
print("\nComputing Silhouette Scores (k = 2 … 14) …")
sil_scores = []
sil_range  = range(2, 15)

for k in sil_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    score  = silhouette_score(X_scaled, labels)
    sil_scores.append(score)
    print(f"  k={k:2d}  silhouette={score:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(list(sil_range), sil_scores, color=["#FF6B35" if k == BEST_K else "#1DB954" for k in sil_range],
       edgecolor="white", linewidth=0.8)
ax.set_xlabel("Number of Clusters  k", fontsize=13)
ax.set_ylabel("Silhouette Score", fontsize=13)
ax.set_title("Silhouette Score per k  (higher = more compact clusters)", fontsize=14, fontweight="bold")
ax.set_xticks(list(sil_range))

# Annotate the chosen k
idx = list(sil_range).index(BEST_K)
ax.annotate(f"Chosen k={BEST_K}\n({sil_scores[idx]:.4f})",
            xy=(BEST_K, sil_scores[idx]),
            xytext=(BEST_K + 0.5, sil_scores[idx] + 0.015),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "03_silhouette_scores.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── 4. Final K-Means Model ────────────────────────────────────────────────
print(f"\nFitting final K-Means with k = {BEST_K} …")
km_final = KMeans(n_clusters=BEST_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
labels   = km_final.fit_predict(X_scaled)
df["cluster"] = labels
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

print(f"\nFinal inertia     : {km_final.inertia_:,.0f}")
sil_final = silhouette_score(X_scaled, labels)
print(f"Final silhouette  : {sil_final:.4f}")

print("\nCluster sizes:")
counts = df.groupby(["cluster", "cluster_name"]).size().reset_index(name="count")
for _, row in counts.iterrows():
    bar = "█" * (row["count"] // 50)
    print(f"  Cluster {row['cluster']} — {row['cluster_name']:<25}  {row['count']:>5} songs  {bar}")

# ─── 5. Cluster Centroids ──────────────────────────────────────────────────
centroids_scaled = pd.DataFrame(km_final.cluster_centers_, columns=FEATURES)
centroids_raw    = pd.DataFrame(scaler.inverse_transform(km_final.cluster_centers_), columns=FEATURES)
centroids_raw.index = [CLUSTER_NAMES[i] for i in range(BEST_K)]

print("\nCluster centroids (raw feature values):")
print(centroids_raw.round(3).to_string())

# ─── 6. Save labelled dataset ──────────────────────────────────────────────
out_csv = os.path.join(OUTPUT_DIR, "songs_with_clusters.csv")
df.to_csv(out_csv, index=False)
print(f"\n[Saved] {out_csv}")

print("\n✅  K-Means clustering complete.")
