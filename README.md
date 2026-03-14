# ISA Mini-Project 1: Sequential Recommendation on Spotify MPD

*Adam Candrák, Tomáš Kubričan*

## Overview

Sequential/session-based music recommendation system using a \[we dont know yet\] model trained on the Spotify Million Playlist Dataset. The model learns to predict the next track in a playlist given the sequence of preceding tracks.

**Course:** Intelligent System Applications (ISA) \
**School year:** 2025/2026 \
**Cookie:** 4 – Sequential/Session-Based Models \
**Framework:** asi PyTorch, *ok Tomáš?*

## Dataset

The dataset contains one million playlists, where each playlist contains information about the playlist itself (e.g., playlist title, duration) and each track it contains (e.g., track id, artist id, track name, duration). The dataset was used as part of a “playlist continuation” challenge: given a playlist name and a few tracks, predict which other tracks would fit best with that playlist.
Source of the dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   └── *.json                 # Raw MPD slices (mpd.slice.*.json)
├── notebooks/
⋮
├── src/
⋮
├── models/                    # Saved model checkpoints
└── figures/                   # Plots and visualizations for the report
```

## Quickstart



## Notebooks





## Evaluation Metrics

