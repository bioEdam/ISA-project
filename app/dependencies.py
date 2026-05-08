import asyncpg
from fastapi import HTTPException, Request


def get_pool(request: Request) -> asyncpg.Pool:
    pool = request.app.state.db_pool
    if pool is None:
        raise HTTPException(503, detail="Database not configured (DATABASE_URL missing)")
    return pool


def get_demo(request: Request):
    return request.app.state.demo
