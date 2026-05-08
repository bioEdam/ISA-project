from pydantic import BaseModel


class RecommendRequest(BaseModel):
    seed_idxs: list[int]
    k: int = 10


class TrackIn(BaseModel):
    track_uri: str
    track_name: str | None = None
    artist_name: str | None = None
    album_name: str | None = None
    duration_ms: int | None = None
    is_seed: bool = True


class PlaylistIn(BaseModel):
    name: str
    tracks: list[TrackIn]


class PlaylistRename(BaseModel):
    name: str
