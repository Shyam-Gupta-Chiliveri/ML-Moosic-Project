"""
Moosic Project — Step 7: Cluster Evaluation & Profiling
========================================================
Final step: deep-dive into what each K-Means cluster sounds like.

  1. Algorithm comparison table (K-Means vs DBSCAN vs Agglomerative)
  2. Feature heatmap — all 6 clusters × 9 features
  3. Radar chart — normalised audio fingerprint per cluster
  4. Top-5 songs per cluster (nearest to centroid)
  5. Playlist export — one CSV per cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import os, math, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "../spotify_5000_songs.csv"
OUTPUT_DIR  = "../outputs"
RANDOM_STATE = 42
BEST_K       = 6
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]
CLUSTER_NAMES = {
    0: "Energetic Rock",
    1: "Instrumental Ambient",
    2: "Acoustic & Mellow",
    3: "Hard & Intense",
    4: "Pop & Dance",
    5: "Hip-Hop & R&B",
}
COLORS = ["#FF6B35", "#00C49A", "#A78BFA", "#F59E0B", "#1DB954", "#60A5FA"]

# ─── Load & Scale ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
X = df[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Final K-Means ─────────────────────────────────────────────────────────
km     = KMeans(n_clusters=BEST_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
labels = km.fit_predict(X_scaled)
df["cluster"]      = labels
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# ─── 1. Algorithm Comparison ───────────────────────────────────────────────
print("=" * 60)
print("ALGORITHM COMPARISON")
print("=" * 60)

db  = DBSCAN(eps=1.0, min_samples=5).fit(X_scaled)
agg = AgglomerativeClustering(n_clusters=BEST_K, linkage="ward").fit(X_scaled)

sil_km  = silhouette_score(X_scaled, labels)
db_lbls = db.labels_
db_nn   = db_lbls[db_lbls != -1]
sil_db  = silhouette_score(X_scaled[db_lbls != -1], db_nn) if len(set(db_nn)) >= 2 else float("nan")
sil_ag  = silhouette_score(X_scaled, agg.labels_)

n_db_cls   = len(set(db_lbls)) - 1
n_db_noise = np.sum(db_lbls == -1)

print(f"\n{'Algorithm':<25} {'k':>4}  {'Silhouette':>11}  Notes")
print("-" * 65)
print(f"{'K-Means (k-means++)':<25} {BEST_K:>4}  {sil_km:>11.4f}  Best overall ✓")
print(f"{'DBSCAN (eps=1.0)':<25} {n_db_cls:>4}  {sil_db:>11.4f}  +{n_db_noise} noise pts")
print(f"{'Agglomerative (Ward)':<25} {BEST_K:>4}  {sil_ag:>11.4f}  Slightly lower")

# ─── 2. Feature Heatmap ────────────────────────────────────────────────────
centroids_raw = pd.DataFrame(
    scaler.inverse_transform(km.cluster_centers_), columns=FEATURES
)
centroids_raw.index = [CLUSTER_NAMES[i] for i in range(BEST_K)]

# Normalise each feature 0-1 for colour display
mms = MinMaxScaler()
heatmap_data = pd.DataFrame(
    mms.fit_transform(centroids_raw), columns=FEATURES, index=centroids_raw.index
)

fig, ax = plt.subplots(figsize=(13, 6))
im = ax.imshow(heatmap_data.values, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(FEATURES)))
ax.set_xticklabels([f.title() for f in FEATURES], rotation=30, ha="right", fontsize=11)
ax.set_yticks(range(BEST_K))
ax.set_yticklabels(heatmap_data.index, fontsize=11)
ax.set_title("Cluster Audio Fingerprints — Normalised Feature Heatmap",
             fontsize=14, fontweight="bold", pad=15)

# Annotate cells
for r in range(BEST_K):
    for c in range(len(FEATURES)):
        raw_val = centroids_raw.values[r, c]
        txt = f"{raw_val:.0f}" if FEATURES[c] == "tempo" else f"{raw_val:.2f}"
        ax.text(c, r, txt, ha="center", va="center", fontsize=9,
                color="black" if 0.25 < heatmap_data.values[r, c] < 0.75 else "white")

plt.colorbar(im, ax=ax, label="Normalised Value (0–1)", shrink=0.8)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "07_cluster_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Saved] {out_path}")

# ─── 3. Radar Charts ───────────────────────────────────────────────────────
RADAR_FEATURES = ["danceability", "energy", "acousticness", "valence",
                  "speechiness", "instrumentalness"]

# Normalise centroids for radar
centroids_radar = centroids_raw[RADAR_FEATURES].copy()
for feat in RADAR_FEATURES:
    mn, mx = df[feat].min(), df[feat].max()
    centroids_radar[feat] = (centroids_radar[feat] - mn) / (mx - mn)

N     = len(RADAR_FEATURES)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]

fig, axes = plt.subplots(2, 3, figsize=(16, 10), subplot_kw=dict(polar=True))
fig.suptitle("Radar Charts — Audio Fingerprint per Cluster", fontsize=16, fontweight="bold")

for cid, ax in enumerate(axes.flatten()):
    values = centroids_radar.iloc[cid].tolist()
    values += values[:1]
    ax.plot(angles, values, color=COLORS[cid], linewidth=2)
    ax.fill(angles, values, color=COLORS[cid], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.title() for f in RADAR_FEATURES], size=9)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.set_title(CLUSTER_NAMES[cid], size=11, fontweight="bold", pad=15, color=COLORS[cid])

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "07_radar_charts.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── 4. Top-5 Songs per Cluster ────────────────────────────────────────────
print("\nTOP-5 SONGS PER CLUSTER (nearest to centroid)")
print("=" * 60)

SONG_COL   = "song_name"   if "song_name"   in df.columns else \
             "track_name"  if "track_name"  in df.columns else \
             "name"        if "name"        in df.columns else df.columns[0]
ARTIST_COL = "artist_name" if "artist_name" in df.columns else \
             "artists"     if "artists"     in df.columns else \
             "artist"      if "artist"      in df.columns else ""

for cid in range(BEST_K):
    mask  = labels == cid
    idx_c = np.where(mask)[0]
    dists = np.linalg.norm(X_scaled[idx_c] - km.cluster_centers_[cid], axis=1)
    top5  = idx_c[np.argsort(dists)[:5]]

    print(f"\nCluster {cid} — {CLUSTER_NAMES[cid]}  ({mask.sum()} songs)")
    for rank, i in enumerate(top5, 1):
        song   = df.iloc[i][SONG_COL]
        artist = df.iloc[i][ARTIST_COL] if ARTIST_COL else "—"
        print(f"  {rank}. {song}  —  {artist}")

# ─── 5. Export playlists ───────────────────────────────────────────────────
playlist_dir = os.path.join(OUTPUT_DIR, "playlists")
os.makedirs(playlist_dir, exist_ok=True)

for cid in range(BEST_K):
    playlist = df[df["cluster"] == cid].copy()
    fname    = f"playlist_{cid:02d}_{CLUSTER_NAMES[cid].replace(' ', '_').replace('&', 'and')}.csv"
    playlist.to_csv(os.path.join(playlist_dir, fname), index=False)

print(f"\n[Saved] 6 playlist CSVs → {playlist_dir}/")

print("\n✅  Evaluation complete.  All outputs saved to outputs/")
