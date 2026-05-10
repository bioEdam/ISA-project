import asyncpg
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_pool
from app.models import PlaylistIn, PlaylistRename, PlaylistTracksUpdate

router = APIRouter(prefix="/api/playlists", tags=["playlists"])


@router.get("")
async def list_playlists(page: int = 1, per_page: int = 20, pool: asyncpg.Pool = Depends(get_pool)):
    offset = (page - 1) * per_page
    total = await pool.fetchval("SELECT COUNT(*) FROM playlists")
    rows = await pool.fetch(
        "SELECT id, name, created_at FROM playlists ORDER BY created_at DESC LIMIT $1 OFFSET $2",
        per_page, offset,
    )
    pages = max(1, (total + per_page - 1) // per_page)
    return {"items": [dict(r) for r in rows], "total": total, "page": page, "pages": pages}


@router.post("", status_code=201)
async def create_playlist(body: PlaylistIn, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        async with conn.transaction():
            pl_id = await conn.fetchval(
                "INSERT INTO playlists (name) VALUES ($1) RETURNING id", body.name
            )
            for i, t in enumerate(body.tracks):
                await conn.execute(
                    """INSERT INTO playlist_tracks
                       (playlist_id, position, track_uri, track_name, artist_name, album_name, duration_ms, is_seed)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    pl_id, i, t.track_uri, t.track_name, t.artist_name,
                    t.album_name, t.duration_ms, t.is_seed,
                )
    return {"id": pl_id}


@router.get("/{playlist_id}")
async def get_playlist(playlist_id: int, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        pl = await conn.fetchrow(
            "SELECT id, name, created_at FROM playlists WHERE id = $1", playlist_id
        )
        if pl is None:
            raise HTTPException(404, "Playlist not found")
        tracks = await conn.fetch(
            "SELECT position, track_uri, track_name, artist_name, album_name, duration_ms, is_seed"
            " FROM playlist_tracks WHERE playlist_id = $1 ORDER BY position",
            playlist_id,
        )
    return {
        "id": pl["id"],
        "name": pl["name"],
        "created_at": pl["created_at"],
        "tracks": [dict(t) for t in tracks],
    }


@router.patch("/{playlist_id}")
async def rename_playlist(playlist_id: int, body: PlaylistRename, pool: asyncpg.Pool = Depends(get_pool)):
    result = await pool.execute(
        "UPDATE playlists SET name = $1, updated_at = NOW() WHERE id = $2",
        body.name, playlist_id,
    )
    if result == "UPDATE 0":
        raise HTTPException(404, "Playlist not found")
    return {"id": playlist_id, "name": body.name}


@router.put("/{playlist_id}/tracks")
async def update_tracks(playlist_id: int, body: PlaylistTracksUpdate, pool: asyncpg.Pool = Depends(get_pool)):
    async with pool.acquire() as conn:
        async with conn.transaction():
            exists = await conn.fetchval("SELECT 1 FROM playlists WHERE id = $1", playlist_id)
            if not exists:
                raise HTTPException(404, "Playlist not found")
            await conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = $1", playlist_id)
            for i, t in enumerate(body.tracks):
                await conn.execute(
                    """INSERT INTO playlist_tracks
                       (playlist_id, position, track_uri, track_name, artist_name, album_name, duration_ms, is_seed)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)""",
                    playlist_id, i, t.track_uri, t.track_name, t.artist_name,
                    t.album_name, t.duration_ms, t.is_seed,
                )
            await conn.execute("UPDATE playlists SET updated_at = NOW() WHERE id = $1", playlist_id)
    return {"id": playlist_id, "track_count": len(body.tracks)}


@router.delete("/{playlist_id}", status_code=204)
async def delete_playlist(playlist_id: int, pool: asyncpg.Pool = Depends(get_pool)):
    result = await pool.execute("DELETE FROM playlists WHERE id = $1", playlist_id)
    if result == "DELETE 0":
        raise HTTPException(404, "Playlist not found")
