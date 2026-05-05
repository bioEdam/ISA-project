# User Manual

## Overview

The GRU Music Recommender is a web application that predicts the next tracks for a playlist based on a sequence of seed tracks. It uses a GRU (Gated Recurrent Unit) neural network trained on 1 million Spotify playlists with a vocabulary of 100,000 tracks.

## Getting Started

1. Open the application in your browser at `http://localhost:8000`
2. The interface has two main areas:
   - **Left panel:** Search and browse tracks
   - **Right panel:** Your seed playlist and recommendations

## How to Use

### Step 1: Find Tracks

**Search by name or artist:**
- Type a track name or artist name in the search bar
- Results appear automatically as you type
- Press Enter to search immediately

**Browse popular tracks:**
- Click the "Browse Popular" button to see the most popular tracks in the dataset

### Step 2: Build Your Seed Playlist

- Click the **"+ Add"** button next to any track to add it to your seed playlist
- The seed playlist appears on the right side of the screen
- A **context quality indicator** shows how much context the model has:
  - **Weak context** (1-2 tracks): Recommendations may be generic
  - **Good context** (3-10 tracks): Recommendations are reasonable
  - **Excellent context** (11+ tracks): Best recommendation quality
- Remove individual tracks by clicking the **X** button
- Click **"Clear"** to remove all tracks from the seed playlist

### Step 3: Get Recommendations

1. Choose how many recommendations you want using the **Top-K** dropdown (5, 10, or 20)
2. Click **"Get Recommendations"**
3. The recommended next tracks appear below, ranked by the model's confidence

## Tips for Better Results

- **Add more seed tracks:** The model performs best with 5-10+ seed tracks that represent a coherent playlist theme
- **Track order matters:** The model is sequential — it considers the order of tracks in your seed playlist. The most recent tracks have the strongest influence
- **Stay within the vocabulary:** The model knows the top 100,000 most popular tracks from the Spotify Million Playlist Dataset. Very obscure tracks may not be recognized
- **Genre consistency:** Seed tracks from the same genre or mood produce more focused recommendations

## Understanding the Results

Each recommendation shows:
- **Rank (#1, #2, ...):** Position in the recommendation list, ordered by model confidence
- **Track name:** The recommended song title
- **Artist name:** The performing artist

## Limitations

- The model was trained on data from the Spotify Million Playlist Dataset (2018). It does not know tracks released after this date.
- Only the top 100,000 most popular tracks are in the vocabulary. Niche or obscure tracks may not appear in search results or recommendations.
- The model predicts based on sequential co-occurrence patterns in playlists. It does not use audio features, lyrics, or explicit genre labels.
