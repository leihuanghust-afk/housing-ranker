"""
app.py — House Ranker Scorer (FastAPI)

Drop this file in your repo as app.py (or replace your existing app.py),
then make sure your Render start command points to it, e.g.:

  uvicorn app:app --host 0.0.0.0 --port $PORT

This version fixes Option A properly:
- /score accepts {"prefs": {...}, "candidates": [ ... ]} where candidates is a LIST
- returns ranked list with per-dimension scores and rank
- robust to missing optional fields (trend/neighborhood/url/etc.)

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import FastAPI
from pydantic import BaseModel, Field


# ----------------------------
# Pydantic request/response models
# ----------------------------

class Trend(BaseModel):
    trend_yoy: Optional[float] = None


class Candidate(BaseModel):
    id: str
    address: Optional[str] = None
    url: Optional[str] = None

    price: Optional[float] = Field(default=None, ge=0)
    beds: Optional[int] = Field(default=None, ge=0)
    commute_min: Optional[int] = Field(default=None, ge=0)

    trend: Optional[Trend] = None
    neighborhood: Optional[Dict[str, Any]] = None


class Weights(BaseModel):
    # You can tune defaults here. They will be normalized to sum to 1.
    commute: float = 0.35
    trend: float = 0.25
    value: float = 0.25
    neighborhood: float = 0.15


class Prefs(BaseModel):
    commute_max_min: Optional[int] = Field(default=None, ge=0)
    budget_max: Optional[float] = Field(default=None, ge=0)
    beds_min: Optional[int] = Field(default=None, ge=0)

    # Optional override; if absent, defaults above are used
    weights: Optional[Weights] = None


class ScoreRequest(BaseModel):
    prefs: Prefs
    candidates: List[Candidate]


class Scores(BaseModel):
    total: float
    commute: float
    trend: float
    value: float
    neighborhood: float


class RankedCandidate(BaseModel):
    id: str
    address: Optional[str] = None
    url: Optional[str] = None

    price: Optional[float] = None
    beds: Optional[int] = None
    commute_min: Optional[int] = None

    scores: Scores
    rank: int


# ----------------------------
# App
# ----------------------------

app = FastAPI(title="House Ranker Scorer", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}


# ----------------------------
# Scoring helpers
# ----------------------------

def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _normalize_weights(w: Weights) -> Dict[str, float]:
    raw = {
        "commute": float(w.commute),
        "trend": float(w.trend),
        "value": float(w.value),
        "neighborhood": float(w.neighborhood),
    }
    s = sum(raw.values())
    if s <= 0:
        # fallback to equal weights
        return {"commute": 0.25, "trend": 0.25, "value": 0.25, "neighborhood": 0.25}
    return {k: v / s for k, v in raw.items()}


def _score_commute(commute_min: Optional[int], commute_max_min: Optional[int]) -> float:
    # If no preference or no data, neutral
    if commute_min is None or commute_max_min is None or commute_max_min <= 0:
        return 0.5
    # Linear decay: <= max => 1.0, 2x max => 0.0
    x = 1.0 - (commute_min / (2.0 * commute_max_min))
    return _clamp01(x)


def _score_value(price: Optional[float], budget_max: Optional[float]) -> float:
    if price is None or budget_max is None or budget_max <= 0:
        return 0.5
    # If price <= budget => 1.0 .. if price >= 1.5*budget => 0.0
    ratio = price / budget_max
    x = 1.0 - ((ratio - 1.0) / 0.5)  # ratio=1 =>1, ratio=1.5=>0
    return _clamp01(x)


def _score_trend(trend: Optional[Trend]) -> float:
    # If no data, neutral
    if trend is None or trend.trend_yoy is None:
        return 0.5
    # Map yoy to 0..1 with a gentle curve:
    # yoy = 0 -> 0.5, yoy = +10 -> ~1, yoy = -10 -> ~0
    yoy = float(trend.trend_yoy)
    x = 0.5 + (yoy / 20.0)
    return _clamp01(x)


def _score_neighborhood(neighborhood: Optional[Dict[str, Any]]) -> float:
    # Placeholder: if you later add real neighborhood features, implement here.
    # If the input provides an explicit score, respect it:
    if isinstance(neighborhood, dict):
        val = neighborhood.get("score")
        if isinstance(val, (int, float)):
            return _clamp01(float(val))
    return 0.5


def _beds_gate(beds: Optional[int], beds_min: Optional[int]) -> bool:
    # If no prefs or no data, allow
    if beds_min is None or beds_min <= 0 or beds is None:
        return True
    return beds >= beds_min


# ----------------------------
# Main endpoint
# ----------------------------

@app.post("/score", response_model=List[RankedCandidate])
def score(req: ScoreRequest) -> List[RankedCandidate]:
    prefs = req.prefs
    weights = _normalize_weights(prefs.weights or Weights())

    ranked: List[Dict[str, Any]] = []

    for c in req.candidates:
        # Optional gate on beds_min
        if not _beds_gate(c.beds, prefs.beds_min):
            # You can either skip or heavily penalize. Here we skip.
            continue

        s_commute = _score_commute(c.commute_min, prefs.commute_max_min)
        s_value = _score_value(c.price, prefs.budget_max)
        s_trend = _score_trend(c.trend)
        s_neigh = _score_neighborhood(c.neighborhood)

        total = (
            weights["commute"] * s_commute
            + weights["value"] * s_value
            + weights["trend"] * s_trend
            + weights["neighborhood"] * s_neigh
        )

        ranked.append({
            "id": c.id,
            "address": c.address,
            "url": c.url,
            "price": c.price,
            "beds": c.beds,
            "commute_min": c.commute_min,
            "scores": {
                "total": round(float(total), 6),
                "commute": round(float(s_commute), 6),
                "trend": round(float(s_trend), 6),
                "value": round(float(s_value), 6),
                "neighborhood": round(float(s_neigh), 6),
            },
        })

    # Deterministic sort: total desc, then commute asc (if present), then price asc (if present), then id
    def sort_key(x: Dict[str, Any]):
        total = x["scores"]["total"]
        commute = x.get("commute_min")
        price = x.get("price")
        # commute/price may be None; treat None as "worst" for sorting tie-breakers
        commute_key = commute if isinstance(commute, (int, float)) else 10**9
        price_key = price if isinstance(price, (int, float)) else 10**18
        return (-total, commute_key, price_key, str(x.get("id", "")))

    ranked.sort(key=sort_key)

    # Assign ranks
    out: List[RankedCandidate] = []
    for i, x in enumerate(ranked, start=1):
        out.append(RankedCandidate(
            id=x["id"],
            address=x.get("address"),
            url=x.get("url"),
            price=x.get("price"),
            beds=x.get("beds"),
            commute_min=x.get("commute_min"),
            scores=Scores(**x["scores"]),
            rank=i,
        ))

    return out
