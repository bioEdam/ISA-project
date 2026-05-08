from fastapi import APIRouter, Depends

from app.dependencies import get_demo
from app.models import RecommendRequest

router = APIRouter(prefix="/api", tags=["recommender"])


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/search")
async def search(q: str, max_results: int = 20, demo=Depends(get_demo)):
    return demo.search(q, max_results=max_results)


@router.get("/top")
async def top(n: int = 20, demo=Depends(get_demo)):
    return demo.top_popular(n)


@router.post("/recommend")
async def recommend(body: RecommendRequest, demo=Depends(get_demo)):
    return demo.recommend(body.seed_idxs, k=body.k)
