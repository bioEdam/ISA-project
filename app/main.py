import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

from recommender import GRUDemo
from app.routers import recommender, playlists


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.demo = GRUDemo(root=ROOT)
    url = os.environ.get("DATABASE_URL")
    app.state.db_pool = await asyncpg.create_pool(url) if url else None
    yield
    if app.state.db_pool:
        await app.state.db_pool.close()


class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["cache-control"] = "no-store"
        return response


app = FastAPI(title="GRU Music Recommender", lifespan=lifespan)
app.add_middleware(NoCacheStaticMiddleware)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

app.include_router(recommender.router)
app.include_router(playlists.router)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
