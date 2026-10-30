"""
Moosic Project — Step 5: DBSCAN Clustering
===========================================
DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
identifies arbitrarily-shaped clusters based on density, without requiring
k to be specified upfront.

Key parameters:
  eps         — neighbourhood radius (found via k-distance graph)
  min_samples — minimum points to form a dense region (default: 5)

Findings:
  At eps = 1.0, DBSCAN finds 7 clusters but 931 songs (~18%) are labelled
  as noise (-1).  The silhouette score (−0.022) reveals that the audio
  feature space is largely one continuous distribution — confirming that
  soft, centroid-based K-Means boundaries are the right tool here.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "../spotify_5000_songs.csv"
OUTPUT_DIR  = "../outputs"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]

# ─── Load & Scale ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
X = df[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── K-Distance Graph (eps selection) ─────────────────────────────────────
print("Building k-distance graph (k=5) to find optimal eps …")
k = 5
nbrs    = NearestNeighbors(n_neighbors=k).fit(X_scaled)
dists, _ = nbrs.kneighbors(X_scaled)
k_dists  = np.sort(dists[:, k - 1])[::-1]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(k_dists, color="#1DB954", linewidth=1.5)
ax.axhline(1.0, color="#FF6B35", linewidth=2, linestyle="--", label="eps = 1.0 (chosen)")
ax.set_xlabel("Points (sorted by distance)", fontsize=12)
ax.set_ylabel(f"{k}-th Nearest Neighbour Distance", fontsize=12)
ax.set_title("K-Distance Graph — Choosing DBSCAN eps", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "05_kdistance.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── DBSCAN parameter sweep ────────────────────────────────────────────────
eps_values = [0.5, 0.8, 1.0, 1.2, 1.5]
print(f"\n{'eps':>6}  {'clusters':>9}  {'noise':>7}  {'noise%':>7}  {'silhouette':>11}")
print("-" * 50)

results = {}
for eps in eps_values:
    db     = DBSCAN(eps=eps, min_samples=5).fit(X_scaled)
    lbls   = db.labels_
    n_cls  = len(set(lbls)) - (1 if -1 in lbls else 0)
    n_ns   = np.sum(lbls == -1)
    noise_pct = n_ns / len(lbls) * 100

    unique_non_noise = len(set(lbls[lbls != -1]))
    if unique_non_noise >= 2:
        sil = silhouette_score(X_scaled[lbls != -1], lbls[lbls != -1])
    else:
        sil = float("nan")

    results[eps] = dict(n_cls=n_cls, n_noise=n_ns, noise_pct=noise_pct, sil=sil, labels=lbls)
    print(f"{eps:>6.1f}  {n_cls:>9}  {n_ns:>7}  {noise_pct:>6.1f}%  {sil:>11.4f}")

# ─── Best result detail (eps=1.0) ─────────────────────────────────────────
best_eps = 1.0
r        = results[best_eps]
print(f"\nSelected eps = {best_eps}:")
print(f"  Clusters found : {r['n_cls']}")
print(f"  Noise points   : {r['n_noise']} ({r['noise_pct']:.1f}%)")
print(f"  Silhouette     : {r['sil']:.4f}")

# PCA for visualisation
from sklearn.decomposition import PCA
pca    = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca  = pca.fit_transform(X_scaled)
lbls   = r["labels"]

# Map cluster ids to colours
unique_ids = sorted(set(lbls))
cmap  = plt.cm.get_cmap("tab10", len(unique_ids))
color_map = {cid: cmap(i) for i, cid in enumerate(unique_ids)}
color_map[-1] = (0.6, 0.6, 0.6, 0.3)   # grey for noise

fig, ax = plt.subplots(figsize=(12, 8))
for cid in unique_ids:
    mask = lbls == cid
    label = "Noise" if cid == -1 else f"Cluster {cid}"
    size  = 8 if cid == -1 else 20
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=[color_map[cid]], label=label, s=size,
               alpha=0.4, linewidths=0)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)", fontsize=12)
ax.set_title(f"DBSCAN Clusters (eps={best_eps}, min_samples=5)\n"
             f"{r['n_cls']} clusters · {r['n_noise']} noise points ({r['noise_pct']:.1f}%)",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9, markerscale=2)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "05_dbscan_clusters.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

print("""
Key insight:
  DBSCAN's very low silhouette (-0.022) and large noise fraction confirm
  that Spotify's audio feature space is a continuous, overlapping
  distribution without sharp density boundaries between genres.
  This validates K-Means' soft-boundary approach as the better choice
  for music playlist generation.
""")

print("✅  DBSCAN analysis complete.")
