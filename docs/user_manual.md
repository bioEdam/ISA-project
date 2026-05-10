# User Manual

## Overview

The GRU Music Recommender is a web application that predicts the next tracks for a playlist based on a sequence of seed tracks. It uses a GRU (Gated Recurrent Unit) neural network trained on 1 million Spotify playlists with a vocabulary of 100,000 tracks.

## Getting Started

1. Open the application in your browser at `http://localhost:8000`
2. The interface has three main areas:
   - **Left sidebar:** Saved playlists (browse, load, edit, delete)
   - **Middle panel:** Search and browse tracks
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

### Step 4: Save Your Playlist

If the application is running with a database (via Docker Compose), you can save your playlist:

1. Enter a name in the **playlist name** text field (defaults to "My Playlist" if left blank)
2. Click **"Save"**
3. The playlist is saved to the database, including both your seed tracks and any generated recommendations
4. The button changes to **"Saved"** to confirm success, along with a status message showing the playlist ID
5. The new playlist appears in the left sidebar immediately

## Managing Playlists

The left sidebar shows all saved playlists, paginated in groups of 20. Use the **Prev/Next** buttons at the bottom to navigate between pages.

### Loading a Playlist

- Click on any playlist name in the sidebar to load it into the seed area
- The playlist's seed tracks are resolved and populated into your current seed list
- You can then click **"Get Recommendations"** to generate new suggestions based on those tracks
- The loaded playlist is highlighted in the sidebar

### Editing a Playlist

1. Hover over a playlist in the sidebar and click the **pencil icon** to enter edit mode
2. The playlist's tracks are loaded into the seed area and an "Editing: ..." indicator appears
3. The **"Save"** button changes to **"Update"**
4. Make your changes:
   - **Rename:** Change the name in the playlist name text field
   - **Remove tracks:** Click the **X** button next to any track in the seed list
   - **Add tracks:** Use the search panel to find and add new tracks
5. Click **"Update"** to save your changes
6. Click **"Clear"** to exit edit mode without saving

### Deleting a Playlist

1. Hover over a playlist in the sidebar and click the **trash icon**
2. Confirm the deletion in the dialog that appears
3. The playlist is permanently removed from the database

### Notes

- All playlists in the database are editable (including pre-seeded playlists from the training data)
- There is no user management — all playlists are shared
- If a track in a saved playlist is no longer in the model's vocabulary (e.g., after retraining), it will not appear when the playlist is loaded

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
- The save-playlist feature requires a PostgreSQL database connection (available when running via Docker Compose). Without a database, the save button will show an error.
