# src/jailbreak_detection/api/routes.py
"""
Endpoints:
    POST /analyze         one text
    POST /analyze/batch   up to 20 texts
    GET  /health          layer status (503 if a layer is missing)
    GET  /stats           counters since startup

Handlers are plain `def`, not `async def`: model inference is blocking
CPU work, and FastAPI runs plain functions in a thread pool so one slow
request does not hold up the others.
"""

import threading
import time
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from src.jailbreak_detection.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    HealthResponse,
    LayerBreakdownResponse,
    StatsResponse,
)
from src.jailbreak_detection.constants import DECISION_ALLOW, DECISION_BLOCK, DECISION_MONITOR
from src.jailbreak_detection.utils import logger

router = APIRouter()
_stats_lock = threading.Lock()


def get_engine(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="Detection engine not initialized")
    return engine


def get_stats(request: Request) -> dict:
    return request.app.state.stats


def to_response(result, include_breakdown: bool = True) -> AnalyzeResponse:
    layers = LayerBreakdownResponse(**asdict(result.layers)) if include_breakdown else None
    return AnalyzeResponse(
        decision=result.decision,
        risk_score=result.risk_score,
        attack_category=result.attack_category,
        severity=result.severity,
        is_attack=result.is_attack,
        explanation=result.explanation,
        original_text=result.original_text,
        processed_text=result.processed_text,
        override_applied=result.override_applied,
        fast_path=result.fast_path,
        layers=layers,
    )


def count(stats: dict, decision: str):
    with _stats_lock:   # handlers run in parallel threads
        stats["total_analyzed"] += 1
        if decision == DECISION_BLOCK:
            stats["total_blocked"] += 1
        elif decision == DECISION_MONITOR:
            stats["total_monitored"] += 1
        else:
            stats["total_allowed"] += 1


@router.get("/", tags=["info"])
def root():
    return {
        "name": "Jailbreak Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@router.post("/analyze", response_model=AnalyzeResponse, tags=["detection"])
def analyze(body: AnalyzeRequest, engine=Depends(get_engine), stats=Depends(get_stats)):
    """Score one text: decision (BLOCK / MONITOR / ALLOW), risk score, category and per-layer details."""
    try:
        result = engine.analyze(body.text)
    except Exception as e:
        logger.exception(f"/analyze failed: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

    count(stats, result.decision)
    logger.info(f"/analyze {result.decision} risk={result.risk_score:.3f} {result.attack_category}")
    return to_response(result, body.include_breakdown)


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse, tags=["detection"])
def analyze_batch(body: BatchAnalyzeRequest, engine=Depends(get_engine), stats=Depends(get_stats)):
    """Score up to 20 texts (50,000 characters in total)."""
    try:
        results = engine.analyze_batch(body.texts)
    except Exception as e:
        logger.exception(f"/analyze/batch failed: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")

    decisions = {DECISION_BLOCK: 0, DECISION_MONITOR: 0, DECISION_ALLOW: 0}
    for r in results:
        decisions[r.decision] += 1
        count(stats, r.decision)
    attacks = sum(1 for r in results if r.is_attack)

    return BatchAnalyzeResponse(
        results=[to_response(r, body.include_breakdown) for r in results],
        total=len(results),
        attacks_found=attacks,
        benign_count=len(results) - attacks,
        summary=decisions,
    )


@router.get("/health", response_model=HealthResponse, tags=["system"],
            responses={503: {"description": "A detection layer is not loaded"}})
def health(response: Response, engine=Depends(get_engine)):
    """200 when all 3 layers are loaded, 503 otherwise, so Docker notices missing models."""
    status = engine.status()
    if status["layers_active"] < 3:
        response.status_code = 503

    return HealthResponse(
        status="ok" if status["layers_active"] == 3 else "degraded",
        layers_active=status["layers_active"],
        rule_engine=status["rule_engine"],
        ml_classifier=status["ml_classifier"],
        embedding=status["embedding"],
        model_info={
            "classifier": "distilbert-base-uncased (fine-tuned)",
            "embedder": "all-mpnet-base-v2",
            "index_size": engine.embedding.index_size() if engine.embedding_ok else 0,
        },
    )


@router.get("/stats", response_model=StatsResponse, tags=["system"])
def stats(request: Request, counters=Depends(get_stats)):
    """Request counts and block rate since the server started."""
    total = counters["total_analyzed"]
    blocked = counters["total_blocked"]
    return StatsResponse(
        total_analyzed=total,
        total_blocked=blocked,
        total_monitored=counters["total_monitored"],
        total_allowed=counters["total_allowed"],
        block_rate=round(blocked / total, 4) if total else 0.0,
        uptime_seconds=round(time.time() - request.app.state.start_time, 1),
    )
