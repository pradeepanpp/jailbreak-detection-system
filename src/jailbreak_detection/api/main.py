# src/jailbreak_detection/api/main.py
"""
FastAPI application entry point.

Startup sequence:
  1. Load RiskEngine (rule engine + ML classifier + embedding matcher)
  2. Register routes
  3. Serve on 0.0.0.0:8000

Run:
    uvicorn src.jailbreak_detection.api.main:app --reload --port 8000

Or via Python:
    python src/jailbreak_detection/api/main.py

Interactive docs:
    http://localhost:8000/docs    (Swagger UI)
    http://localhost:8000/redoc   (ReDoc)
"""

import sys
import time
import os
sys.path.append(".")

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.jailbreak_detection.api.routes import router
from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
from src.jailbreak_detection.utils import logger


# ─────────────────────────────────────────────
# LIFESPAN — startup / shutdown
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load all models at startup, release at shutdown.
    FastAPI lifespan replaces deprecated @app.on_event("startup").
    """
    # ── STARTUP ───────────────────────────────
    logger.info("=" * 52)
    logger.info("  Jailbreak Detection API — Starting Up")
    logger.info("=" * 52)

    t0 = time.time()
    logger.info("Loading RiskEngine (rule + ML + embedding)...")

    try:
        app.state.engine = RiskEngine(
            load_classifier = True,
            load_embedding  = True,
        )
        status = app.state.engine.status()
        logger.info(f"RiskEngine loaded in {time.time() - t0:.1f}s")
        logger.info(f"Active layers: {status['layers_active']}/3")

        if status["layers_active"] < 3:
            logger.warning(
                "Not all layers loaded. Run train_classifier.py "
                "and build_index.py for full capability."
            )
    except Exception as e:
        logger.error(f"Failed to load RiskEngine: {e}")
        # Still start server — health endpoint will report error
        app.state.engine = None

    # Runtime stats counter
    app.state.stats = {
        "total_analyzed":  0,
        "total_blocked":   0,
        "total_monitored": 0,
        "total_allowed":   0,
    }
    app.state.start_time = time.time()

    logger.info("=" * 52)
    logger.info("  API ready — http://localhost:8000")
    logger.info("  Docs     — http://localhost:8000/docs")
    logger.info("=" * 52)

    yield   # server runs here

    # ── SHUTDOWN ──────────────────────────────
    logger.info("Shutting down — releasing models...")
    app.state.engine = None


# ─────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "Jailbreak Detection API",
        description = (
            "Multi-layer LLM jailbreak and adversarial prompt detection system.\n\n"
            "**Layers:**\n"
            "- Rule Engine: 33 regex patterns for known attacks\n"
            "- ML Classifier: fine-tuned DistilBERT (7-class, F1=0.884)\n"
            "- Embedding Matcher: FAISS semantic similarity (all-mpnet-base-v2)\n\n"
            "**Attack categories detected:**\n"
            "direct_jailbreak, prompt_injection, roleplay_hijack, "
            "encoding_attack, manyshot, indirect_injection"
        ),
        version     = "1.0.0",
        lifespan    = lifespan,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── CORS ──────────────────────────────────
    # Allows the Streamlit dashboard (localhost:8501)
    # to call the API (localhost:8000) without CORS errors
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── REQUEST LOGGING ───────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = round((time.time() - start) * 1000, 1)
        logger.debug(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} ({duration}ms)"
        )
        return response

    # ── GLOBAL EXCEPTION HANDLER ──────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code = 500,
            content     = {
                "error":  "Internal server error",
                "detail": str(exc),
                "code":   500,
            }
        )

    # ── ROUTES ────────────────────────────────
    app.include_router(router)

    return app


# ─────────────────────────────────────────────
# APP INSTANCE
# ─────────────────────────────────────────────

app = create_app()


# ─────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "src.jailbreak_detection.api.main:app",
        host    = host,
        port    = port,
        reload  = True,   # auto-reload on file changes
        workers = 1,      # single worker — models are loaded in memory
        log_level = "warning",  # suppress uvicorn noise, our logger handles it
    )