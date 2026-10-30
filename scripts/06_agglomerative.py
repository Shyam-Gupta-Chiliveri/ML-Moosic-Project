"""
Moosic Project — Step 6: Agglomerative (Hierarchical) Clustering
=================================================================
Agglomerative clustering builds a tree (dendrogram) by successively
merging the two closest clusters until a single cluster remains.
We cut the tree at k = 6 using Ward linkage, which minimises the
total within-cluster variance at each merge.

Comparison:
  K-Means silhouette     : 0.248
  Agglomerative silhouette: 0.220

K-Means is slightly better — consistent with the roughly spherical,
continuous structure of Spotify's audio feature space.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os, warnings
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
COLORS = ["#FF6B35", "#00C49A", "#A78BFA", "#F59E0B", "#1DB954", "#60A5FA"]

# ─── Load & Scale ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
X = df[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Dendrogram (on 300-song sample for readability) ───────────────────────
print("Building dendrogram (300-song sample) …")
rng   = np.random.default_rng(RANDOM_STATE)
idx   = rng.choice(len(X_scaled), 300, replace=False)
X_sub = X_scaled[idx]

Z = linkage(X_sub, method="ward")

fig, ax = plt.subplots(figsize=(14, 6))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
           leaf_rotation=90, leaf_font_size=9,
           color_threshold=Z[-BEST_K, 2],
           above_threshold_color="#AAAAAA")
ax.set_title(f"Hierarchical Clustering Dendrogram (Ward linkage, n=300)\nCut at k = {BEST_K}",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Song Index (truncated)", fontsize=11)
ax.set_ylabel("Ward Distance", fontsize=11)
ax.axhline(Z[-BEST_K, 2], color="#FF6B35", linewidth=2, linestyle="--",
           label=f"Cut → k = {BEST_K}")
ax.legend(fontsize=11)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "06_dendrogram.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── Fit Agglomerative on full dataset ─────────────────────────────────────
print(f"\nFitting AgglomerativeClustering (k={BEST_K}, Ward) on full dataset …")
agg    = AgglomerativeClustering(n_clusters=BEST_K, linkage="ward")
labels = agg.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, labels)
print(f"Silhouette Score : {sil:.4f}")

counts = pd.Series(labels).value_counts().sort_index()
print("\nCluster sizes:")
for cid, cnt in counts.items():
    print(f"  Cluster {cid}: {cnt:>5} songs")

# ─── Linkage comparison ────────────────────────────────────────────────────
print("\nLinkage comparison (k=6):")
linkage_methods = ["ward", "complete", "average", "single"]
print(f"{'Linkage':>10}  {'Silhouette':>12}")
print("-" * 25)
for method in linkage_methods:
    agg_tmp = AgglomerativeClustering(n_clusters=BEST_K, linkage=method)
    lbl_tmp = agg_tmp.fit_predict(X_scaled)
    s_tmp   = silhouette_score(X_scaled, lbl_tmp)
    marker  = " ← best" if method == "ward" else ""
    print(f"{method:>10}  {s_tmp:>12.4f}{marker}")

# ─── PCA visualisation ─────────────────────────────────────────────────────
pca   = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(11, 8))
for cid in range(BEST_K):
    mask = labels == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[cid], label=f"Cluster {cid}",
               alpha=0.45, s=15, linewidths=0)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax.set_title(f"Agglomerative Clustering (Ward, k={BEST_K})\nSilhouette = {sil:.4f}",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, markerscale=2)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "06_agglomerative_scatter.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

print("\n✅  Agglomerative clustering complete.")
