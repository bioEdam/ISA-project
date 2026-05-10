CREATE TABLE playlists (
    id          SERIAL       PRIMARY KEY,
    name        VARCHAR(255) NOT NULL,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE playlist_tracks (
    id          SERIAL       PRIMARY KEY,
    playlist_id INT          NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    position    INT          NOT NULL,
    track_uri   VARCHAR(255) NOT NULL,
    track_name  VARCHAR(500),
    artist_name VARCHAR(500),
    album_name  VARCHAR(500),
    duration_ms INT,
    is_seed     BOOLEAN      NOT NULL DEFAULT TRUE,
    UNIQUE (playlist_id, position)
);

CREATE INDEX idx_pt_playlist ON playlist_tracks(playlist_id);

CREATE TABLE retrain_log (
    id              SERIAL       PRIMARY KEY,
    started_at      TIMESTAMPTZ  NOT NULL,
    finished_at     TIMESTAMPTZ,
    num_playlists   INT          NOT NULL,
    num_sequences   INT,
    avg_loss        FLOAT,
    status          VARCHAR(20)  NOT NULL DEFAULT 'running',
    error_message   TEXT
);
