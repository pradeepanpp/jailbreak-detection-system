# src/jailbreak_detection/api/main.py
"""
FastAPI app.

    uvicorn src.jailbreak_detection.api.main:app --port 8000
    docs: http://localhost:8000/docs
"""

import os
import sys
import time
from contextlib import asynccontextmanager

sys.path.append(".")

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.jailbreak_detection.aggregator.risk_engine import RiskEngine
from src.jailbreak_detection.api.routes import router
from src.jailbreak_detection.utils import logger

DESCRIPTION = (
    "Detects jailbreak and prompt-injection attempts with three layers: "
    "regex rules, a fine-tuned DistilBERT classifier and FAISS similarity "
    "to known attacks."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the models once at startup."""
    start = time.time()
    try:
        engine = RiskEngine(load_classifier=True, load_embedding=True)
        engine.warm_up()   # so the first real request does not pay the loading cost
        app.state.engine = engine
        layers = engine.status()["layers_active"]
        logger.info(f"Engine ready in {time.time() - start:.1f}s with {layers}/3 layers")
        if layers < 3:
            logger.warning("Not all layers loaded: run train_classifier.py and build_index.py")
    except Exception as e:
        logger.error(f"Failed to load the engine: {e}")
        app.state.engine = None   # the server still starts; /health reports the problem

    app.state.stats = {"total_analyzed": 0, "total_blocked": 0, "total_monitored": 0, "total_allowed": 0}
    app.state.start_time = time.time()
    yield
    app.state.engine = None


def create_app() -> FastAPI:
    app = FastAPI(title="Jailbreak Detection API", description=DESCRIPTION,
                  version="1.0.0", lifespan=lifespan)

    # Only listed origins may call the API from a browser (default: the dashboard)
    origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:8501").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        logger.debug(f"{request.method} {request.url.path} -> {response.status_code} "
                     f"({(time.time() - start) * 1000:.1f}ms)")
        return response

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        # Details go to the log; the client gets a generic message
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content={"error": "Internal server error", "code": 500})

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.jailbreak_detection.api.main:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
        workers=1,   # models live in memory, one worker
        log_level="warning",
    )
