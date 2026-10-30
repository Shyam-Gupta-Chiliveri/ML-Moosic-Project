# 🎵 Moosic Atlas — ML Music Clustering Project

An interactive music clustering dashboard built with K-Means (K=5) on 5,000 Spotify tracks.

## 🌐 Live Dashboard

**[→ Open Moosic Atlas Dashboard](https://shyam-gupta-chiliveri.github.io/ML-Moosic-Project/moosic_atlas_k5.html)**

Share this link — it opens in any browser, no login needed.

---

## Project Overview

This project applies unsupervised machine learning to discover natural groupings ("playlists") in Spotify audio data using 9 audio features:

| Feature | Description |
|---|---|
| Danceability | How suitable for dancing (0–1) |
| Energy | Intensity and activity (0–1) |
| Acousticness | Acoustic confidence (0–1) |
| Tempo | Beats per minute |
| Valence | Musical positiveness (0–1) |
| Speechiness | Spoken word presence (0–1) |
| Instrumentalness | No vocals confidence (0–1) |
| Liveness | Live performance detection (0–1) |
| Loudness | Overall loudness (dB) |

### 5 Discovered Clusters

| # | Cluster Name | Character |
|---|---|---|
| 0 | Hard & Intense | High energy, aggressive, heavy metal |
| 1 | Chill & Acoustic | Low energy, acoustic, calm |
| 2 | Dance & Pop | High danceability, upbeat |
| 3 | Dark & Atmospheric | Low valence, brooding |
| 4 | Upbeat & Energetic | High valence + energy, feel-good |

---

## Repository Structure

```
ML-Moosic-Project/
├── moosic_atlas_k5.html      ← Interactive dashboard (open this!)
├── notebooks/
│   └── moosic_clustering.ipynb  ← Full analysis notebook
├── scripts/
│   ├── 01_data_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_kmeans_clustering.py
│   ├── 04_dimensionality.py
│   ├── 05_dbscan.py
│   ├── 06_agglomerative.py
│   └── 07_evaluation.py
├── spotify_5000_songs.csv    ← Source dataset
├── atlas_data.js             ← Dashboard data (JS)
├── atlas_data.json           ← Dashboard data (JSON)
└── requirements.txt          ← Python dependencies
```

## Running the Notebook

```bash
pip install -r requirements.txt
jupyter notebook notebooks/moosic_clustering.ipynb
```

## Tech Stack

- **Python** — pandas, scikit-learn, matplotlib, seaborn
- **Algorithm** — K-Means (k-means++, K=5, StandardScaler)
- **Dashboard** — Vanilla HTML/CSS/JS, Canvas API radar charts
- **Data** — Spotify Web API audio features (5,000 tracks)

---

*ML Moosic Project — Data Science Bootcamp*
