# src/jailbreak_detection/api/routes.py
"""
FastAPI route handlers.

All routes use the shared RiskEngine instance
injected via FastAPI dependency injection.

Endpoints:
    POST /analyze           — single text detection
    POST /analyze/batch     — batch detection (up to 50)
    GET  /health            — layer status check
    GET  /stats             — runtime counters
    GET  /                  — welcome / docs redirect
"""

import time
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from src.jailbreak_detection.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchAnalyzeRequest,
    BatchAnalyzeResponse,
    LayerBreakdownResponse,
    HealthResponse,
    StatsResponse,
    ErrorResponse,
)
from src.jailbreak_detection.aggregator.risk_engine import RiskResult
from src.jailbreak_detection.constants import (
    DECISION_BLOCK,
    DECISION_MONITOR,
    DECISION_ALLOW,
)
from src.jailbreak_detection.utils import logger


router = APIRouter()


# ─────────────────────────────────────────────
# DEPENDENCY — get shared engine from app state
# ─────────────────────────────────────────────

def get_engine(request: Request):
    """Dependency injection: pull RiskEngine from app state."""
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Detection engine not initialized"
        )
    return engine


def get_stats(request: Request) -> dict:
    """Dependency injection: pull stats counter from app state."""
    return request.app.state.stats


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def risk_result_to_response(
    result: RiskResult,
    include_breakdown: bool = True
) -> AnalyzeResponse:
    """Convert RiskResult dataclass → AnalyzeResponse Pydantic model."""

    layers_resp = None
    if include_breakdown:
        bd = result.layers
        layers_resp = LayerBreakdownResponse(
            rule_score      = bd.rule_score,
            rule_decision   = bd.rule_decision,
            rule_severity   = bd.rule_severity,
            rule_categories = bd.rule_categories,
            rule_matches    = bd.rule_matches,
            ml_score        = bd.ml_score,
            ml_decision     = bd.ml_decision,
            ml_category     = bd.ml_category,
            ml_confidence   = bd.ml_confidence,
            embedding_score    = bd.embedding_score,
            embedding_decision = bd.embedding_decision,
            embedding_category = bd.embedding_category,
            embedding_sim      = bd.embedding_sim,
            embedding_available= bd.embedding_available,
        )

    return AnalyzeResponse(
        decision        = result.decision,
        risk_score      = result.risk_score,
        attack_category = result.attack_category,
        severity        = result.severity,
        is_attack       = result.is_attack,
        explanation     = result.explanation,
        original_text   = result.original_text,
        processed_text  = result.processed_text,
        override_applied= result.override_applied,
        fast_path       = result.fast_path,
        layers          = layers_resp,
    )


def update_stats(stats: dict, decision: str):
    """Increment runtime counters."""
    stats["total_analyzed"] += 1
    if decision == DECISION_BLOCK:
        stats["total_blocked"] += 1
    elif decision == DECISION_MONITOR:
        stats["total_monitored"] += 1
    else:
        stats["total_allowed"] += 1


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@router.get("/", tags=["info"])
async def root():
    """Welcome message and links to API docs."""
    return {
        "name":        "Jailbreak Detection API",
        "version":     "1.0.0",
        "description": "Multi-layer LLM jailbreak and adversarial prompt detection",
        "docs":        "/docs",
        "redoc":       "/redoc",
        "health":      "/health",
    }


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["detection"],
    summary="Analyze a single text for jailbreak/attack patterns",
    responses={
        200: {"description": "Analysis complete"},
        422: {"description": "Validation error — text too long or blank"},
        503: {"description": "Engine not initialized"},
    }
)
async def analyze(
    body:   AnalyzeRequest,
    engine = Depends(get_engine),
    stats  = Depends(get_stats),
):
    """
    Analyze a single text input through the full detection pipeline.

    - **Rule engine** checks for known attack patterns (fast, deterministic)
    - **ML classifier** predicts attack category using fine-tuned DistilBERT
    - **Embedding matcher** detects semantically similar novel attacks

    Returns a unified risk score, decision (BLOCK/MONITOR/ALLOW),
    attack category, severity (1–5), and per-layer breakdown.
    """
    try:
        result   = engine.analyze(body.text)
        response = risk_result_to_response(result, body.include_breakdown)
        update_stats(stats, result.decision)

        logger.info(
            f"[/analyze] decision={result.decision} "
            f"score={result.risk_score:.3f} "
            f"category={result.attack_category}"
        )

        return response

    except Exception as e:
        logger.error(f"[/analyze] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/analyze/batch",
    response_model=BatchAnalyzeResponse,
    tags=["detection"],
    summary="Analyze multiple texts in a single request (max 50)",
    responses={
        200: {"description": "Batch analysis complete"},
        422: {"description": "Validation error"},
        503: {"description": "Engine not initialized"},
    }
)
async def analyze_batch(
    body:   BatchAnalyzeRequest,
    engine = Depends(get_engine),
    stats  = Depends(get_stats),
):
    """
    Analyze up to 50 texts in a single API call.

    Returns individual results for each text plus a summary
    with aggregate counts (total attacks, benign, BLOCK/MONITOR/ALLOW).
    """
    try:
        risk_results = engine.analyze_batch(body.texts)

        responses = [
            risk_result_to_response(r, body.include_breakdown)
            for r in risk_results
        ]

        # Build summary
        decision_counts = {
            DECISION_BLOCK:   0,
            DECISION_MONITOR: 0,
            DECISION_ALLOW:   0,
        }
        for r in risk_results:
            decision_counts[r.decision] = decision_counts.get(r.decision, 0) + 1
            update_stats(stats, r.decision)

        attacks_found = sum(1 for r in risk_results if r.is_attack)

        logger.info(
            f"[/analyze/batch] {len(body.texts)} texts — "
            f"attacks={attacks_found} "
            f"blocked={decision_counts[DECISION_BLOCK]}"
        )

        return BatchAnalyzeResponse(
            results       = responses,
            total         = len(responses),
            attacks_found = attacks_found,
            benign_count  = len(responses) - attacks_found,
            summary       = decision_counts,
        )

    except Exception as e:
        logger.error(f"[/analyze/batch] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["system"],
    summary="Check system health and layer availability",
)
async def health(engine = Depends(get_engine)):
    """
    Returns the current status of all detection layers.

    - **ok**: all 3 layers active
    - **degraded**: 1-2 layers active (system still functional)
    - **error**: no layers active
    """
    status_info = engine.status()
    layers      = status_info["layers_active"]

    if layers == 3:
        status = "ok"
    elif layers >= 1:
        status = "degraded"
    else:
        status = "error"

    return HealthResponse(
        status        = status,
        layers_active = layers,
        rule_engine   = status_info["rule_engine"],
        ml_classifier = status_info["ml_classifier"],
        embedding     = status_info["embedding"],
        model_info    = {
            "classifier": "distilbert-base-uncased (fine-tuned)",
            "embedder":   "all-mpnet-base-v2",
            "index_size": engine.embedding.index_size() if engine.embedding_ok else 0,
        }
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    tags=["system"],
    summary="Runtime statistics — requests analyzed since startup",
)
async def stats(
    request: Request,
    stats   = Depends(get_stats),
):
    """Returns cumulative request counts and block rate since server start."""
    total    = stats["total_analyzed"]
    blocked  = stats["total_blocked"]
    uptime   = time.time() - request.app.state.start_time

    return StatsResponse(
        total_analyzed  = total,
        total_blocked   = blocked,
        total_monitored = stats["total_monitored"],
        total_allowed   = stats["total_allowed"],
        block_rate      = round(blocked / total, 4) if total > 0 else 0.0,
        uptime_seconds  = round(uptime, 1),
    )