"""
Moosic Project — Step 2: Preprocessing & Scaling
=================================================
Distance-based clustering algorithms (K-Means, DBSCAN, Agglomerative) are
sensitive to feature scale.  A tempo of 180 BPM and a loudness of -5 dB
exist on completely different numeric ranges than danceability (0–1).
Without scaling, tempo would dominate every distance calculation.

This script:
- Selects the 9 audio features used for clustering
- Applies StandardScaler (zero mean, unit variance)
- Verifies the scaled output
- Saves the scaled matrix for downstream steps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH = "../spotify_5000_songs.csv"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]

# ─── 1. Load ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]

X = df[FEATURES].copy()
print(f"Feature matrix shape: {X.shape}")

# ─── 2. Scale ──────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES)

print("\nBefore scaling — feature ranges:")
for feat in FEATURES:
    print(f"  {feat:<20} min={X[feat].min():.3f}  max={X[feat].max():.3f}")

print("\nAfter scaling — feature stats (should be ~mean=0, std=1):")
print(X_scaled_df.describe().loc[["mean", "std"]].round(4).to_string())

# ─── 3. Visual comparison ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Feature Scale: Before vs After StandardScaler", fontsize=15, fontweight="bold")

# Before
X.boxplot(ax=ax1, vert=True, patch_artist=True)
ax1.set_title("Before Scaling", fontsize=13)
ax1.set_ylabel("Raw Value")
ax1.tick_params(axis="x", rotation=45)

# After
X_scaled_df.boxplot(ax=ax2, vert=True, patch_artist=True)
ax2.set_title("After StandardScaler", fontsize=13)
ax2.set_ylabel("Standardised Value (z-score)")
ax2.tick_params(axis="x", rotation=45)
ax2.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.5)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "02_scaling_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Saved] {out_path}")

# ─── 4. Save scaled matrix ─────────────────────────────────────────────────
np.save(os.path.join(OUTPUT_DIR, "X_scaled.npy"), X_scaled)
print(f"[Saved] {os.path.join(OUTPUT_DIR, 'X_scaled.npy')}  (shape: {X_scaled.shape})")

print("\n✅  Preprocessing complete.")
