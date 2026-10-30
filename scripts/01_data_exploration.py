"""
Moosic Project — Step 1: Data Exploration
==========================================
Load the Spotify dataset and perform exploratory data analysis:
- Dataset shape, dtypes, missing values
- Descriptive statistics for audio features
- Distribution plots for each feature
- Correlation heatmap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─── Config ────────────────────────────────────────────────────────────────
DATA_PATH = "../spotify_5000_songs.csv"
OUTPUT_DIR = "../outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURES = [
    "danceability", "energy", "acousticness", "tempo",
    "valence", "speechiness", "instrumentalness", "liveness", "loudness"
]

plt.style.use("seaborn-v0_8-darkgrid")
PALETTE = "#1DB954"   # Spotify green

# ─── 1. Load Data ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]   # strip trailing whitespace

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns          : {list(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows   : {df.duplicated().sum()}")
print(f"\nData types:\n{df.dtypes}")

# ─── 2. Descriptive Statistics ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (audio features)")
print("=" * 60)
print(df[FEATURES].describe().round(4).to_string())

# ─── 3. Distribution Plots ─────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Distribution of Spotify Audio Features", fontsize=18, fontweight="bold", y=1.01)

for ax, feat in zip(axes.flatten(), FEATURES):
    ax.hist(df[feat].dropna(), bins=50, color=PALETTE, edgecolor="white", alpha=0.85)
    ax.set_title(feat.title(), fontsize=12, fontweight="bold")
    ax.set_xlabel("Value", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    mean_val = df[feat].mean()
    ax.axvline(mean_val, color="#FF6B35", linewidth=2, linestyle="--", label=f"Mean: {mean_val:.2f}")
    ax.legend(fontsize=9)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "01_feature_distributions.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Saved] {out_path}")

# ─── 4. Correlation Heatmap ────────────────────────────────────────────────
corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle
cmap = sns.diverging_palette(240, 10, as_cmap=True)

sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmin=-1, vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    ax=ax,
    cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "01_correlation_heatmap.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

# ─── 5. Box Plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Box Plots — Outlier Detection", fontsize=18, fontweight="bold")

for ax, feat in zip(axes.flatten(), FEATURES):
    ax.boxplot(df[feat].dropna(), patch_artist=True,
               boxprops=dict(facecolor=PALETTE, alpha=0.7),
               medianprops=dict(color="white", linewidth=2))
    ax.set_title(feat.title(), fontsize=12)
    ax.set_ylabel("Value")

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "01_boxplots.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"[Saved] {out_path}")

print("\n✅  Data exploration complete — plots saved to outputs/")
