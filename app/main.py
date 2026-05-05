import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

from recommender import GRUDemo

demo: GRUDemo | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global demo
    demo = GRUDemo(root=ROOT)
    yield


app = FastAPI(title="GRU Music Recommender", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


class RecommendRequest(BaseModel):
    seed_idxs: list[int]
    k: int = 10


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/search")
async def search(q: str, max_results: int = 20):
    return demo.search(q, max_results=max_results)


@app.get("/api/top")
async def top(n: int = 20):
    return demo.top_popular(n)


@app.post("/api/recommend")
async def recommend(body: RecommendRequest):
    return demo.recommend(body.seed_idxs, k=body.k)
