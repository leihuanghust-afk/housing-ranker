from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import numpy as np

app = FastAPI(title="House Ranker Scorer", version="0.1.0")

class Prefs(BaseModel):
    commute_max_min: int = 30
    budget_max: Optional[int] = None
    beds_min: Optional[float] = None
    weights: Dict[str, float] = Field(default_factory=lambda: {
        "commute": 0.40,
        "trend": 0.20,
        "value": 0.30,
        "neighborhood": 0.10
    })

class Candidate(BaseModel):
    id: str
    address: str
    price: Optional[int] = None
    beds: Optional[float] = None
    baths: Optional[float] = None
    sqft: Optional[int] = None
    url: Optional[str] = None
    commute_min: Optional[float] = None
    trend: Dict[str, Any] = Field(default_factory=dict)

class ScoreRequest(BaseModel):
    prefs: Prefs
    candidates: List[Candidate]

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))

def score_commute(commute_min: Optional[float], max_min: int) -> float:
    if commute_min is None:
        return 0.0
    if commute_min <= max_min:
        return 1.0
    return clamp01(1.0 - (commute_min - max_min) / 15.0)

def score_value(price: Optional[int], budget_max: Optional[int]) -> float:
    if price is None or budget_max is None:
        return 0.5
    if price <= budget_max:
        return clamp01(0.6 + 0.4 * (budget_max - price) / max(budget_max, 1))
    return clamp01(0.6 - 0.8 * (price - budget_max) / max(budget_max, 1))

def score_trend(trend: Dict[str, Any]) -> float:
    yoy = trend.get("trend_yoy")
    if yoy is None:
        return 0.5
    return clamp01((float(yoy) + 10.0) / 20.0)

@app.post("/score")
def score(req: ScoreRequest):
    prefs = req.prefs
    W = prefs.weights
    keys = ["commute", "trend", "value", "neighborhood"]
    w = np.array([float(W.get(k, 0.0)) for k in keys], dtype=float)
    if w.sum() <= 0:
        w = np.array([0.4, 0.2, 0.3, 0.1], dtype=float)
    w = w / w.sum()

    out = []
    for c in req.candidates:
        if prefs.beds_min is not None and (c.beds is None or c.beds < prefs.beds_min):
            continue

        s_comm = score_commute(c.commute_min, prefs.commute_max_min)
        s_trend = score_trend(c.trend)
        s_value = score_value(c.price, prefs.budget_max)
        s_nei = 0.5

        total = float(w[0]*s_comm + w[1]*s_trend + w[2]*s_value + w[3]*s_nei)

        out.append({
            "id": c.id,
            "address": c.address,
            "url": c.url,
            "price": c.price,
            "commute_min": c.commute_min,
            "scores": {"total": total, "commute": s_comm, "trend": s_trend, "value": s_value, "neighborhood": s_nei}
        })

    out.sort(key=lambda r: r["scores"]["total"], reverse=True)
    for i, r in enumerate(out, start=1):
        r["rank"] = i
    return out
