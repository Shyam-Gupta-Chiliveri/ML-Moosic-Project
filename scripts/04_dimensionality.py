"""
Moosic Project — Step 4: Dimensionality Reduction
==================================================
With 9 features we can't directly visualise cluster separation.
Two complementary techniques project the data to 2D:

  PCA  — linear, preserves global variance structure.
         Fast, deterministic, shows the directions of maximum spread.

  t-SNE — non-linear, preserves local neighbourhood structure.
          Reveals fine-grained cluster topology that PCA hides.
          Run on a 2,000-song sample for speed.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, warnings
warnings.filterwarnings("ignore")

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH   = "../spotify_5000_songs.csv"
OUTPUT_DIR  = "../outputs"
RANDOM_STATE = 42
TSNE_SAMPLE  = 2000
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]
BEST_K = 6
CLUSTER_NAMES = {
    0: "Energetic Rock",
    1: "Instrumental Ambient",
    2: "Acoustic & Mellow",
    3: "Hard & Intense",
    4: "Pop & Dance",
    5: "Hip-Hop & R&B",
}
COLORS = ["#FF6B35", "#00C49A", "#A78BFA", "#F59E0B", "#1DB954", "#60A5FA"]

# ─── Load & fit ────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
X = df[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=BEST_K, init="k-means++", n_init=10, random_state=RANDOM_STATE)
labels = km.fit_predict(X_scaled)
df["cluster"] = labels
df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES)

# ─── PCA ───────────────────────────────────────────────────────────────────
print("Running PCA …")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

var1, var2 = pca.explained_variance_ratio_ * 100
print(f"  PC1 explains {var1:.1f}% of variance")
print(f"  PC2 explains {var2:.1f}% of variance")
print(f"  Total        {var1+var2:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("PCA — 2D Projection of Song Feature Space", fontsize=16, fontweight="bold")

# Scatter
ax = axes[0]
for cid in range(BEST_K):
    mask = labels == cid
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COLORS[cid], label=CLUSTER_NAMES[cid],
               alpha=0.45, s=15, linewidths=0)
ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=12)
ax.set_title("Cluster Scatter", fontsize=13)
ax.legend(loc="upper right", fontsize=9, markerscale=2)

# Loadings (feature arrows)
ax2 = axes[1]
loadings = pca.components_.T
for i, feat in enumerate(FEATURES):
    ax2.arrow(0, 0, loadings[i, 0] * 4, loadings[i, 1] * 4,
              head_width=0.08, head_length=0.05, fc="#1DB954", ec="#1DB954", alpha=0.8)
    ax2.text(loadings[i, 0] * 4.2, loadings[i, 1] * 4.2, feat,
             fontsize=10, ha="center", color="#333333")
ax2.set_xlim(-5, 5)
ax2.set_ylim(-5, 5)
ax2.axhline(0, color="grey", linewidth=0.8, alpha=0.5)
ax2.axvline(0, color="grey", linewidth=0.8, alpha=0.5)
ax2.set_title("PCA Feature Loadings", fontsize=13)
ax2.set_xlabel("PC1", fontsize=12)
ax2.set_ylabel("PC2", fontsize=12)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "04_pca.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── Scree Plot ────────────────────────────────────────────────────────────
pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_) * 100

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_ * 100,
       color="#1DB954", edgecolor="white", label="Individual")
ax.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="#FF6B35", linewidth=2, label="Cumulative")
ax.axhline(80, color="grey", linestyle="--", alpha=0.6, label="80% threshold")
ax.set_xlabel("Principal Component", fontsize=12)
ax.set_ylabel("Explained Variance (%)", fontsize=12)
ax.set_title("PCA Scree Plot", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "04_pca_scree.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── t-SNE ─────────────────────────────────────────────────────────────────
print(f"\nRunning t-SNE on {TSNE_SAMPLE}-song sample …  (this may take ~30 s)")
rng    = np.random.default_rng(RANDOM_STATE)
idx    = rng.choice(len(X_scaled), TSNE_SAMPLE, replace=False)
X_sub  = X_scaled[idx]
lab_sub = labels[idx]

tsne   = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE)
X_tsne = tsne.fit_transform(X_sub)

fig, ax = plt.subplots(figsize=(12, 9))
for cid in range(BEST_K):
    mask = lab_sub == cid
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               c=COLORS[cid], label=CLUSTER_NAMES[cid],
               alpha=0.55, s=20, linewidths=0)
ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
ax.set_title(f"t-SNE Projection — {TSNE_SAMPLE}-Song Sample (perplexity=30)",
             fontsize=14, fontweight="bold")
ax.legend(loc="best", fontsize=10, markerscale=2)
ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "04_tsne.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

print("\n✅  Dimensionality reduction complete.")
