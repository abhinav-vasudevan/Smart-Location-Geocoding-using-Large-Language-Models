"""
FastAPI backend server for the Indian Address Correction System.

Exposes a REST API that accepts raw/noisy Indian address text and returns
corrected, geocoded results by routing through integrated.py (which
dispatches to T5 or LLaMA3 models as appropriate).
"""

import json
import logging
import os
import re
import subprocess
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INTEGRATED_SCRIPT_PATH = os.path.join(CURRENT_DIR, "..", "integrated.py")
SUBPROCESS_TIMEOUT = 240  # seconds

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("address_api")


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
class AddressRequest(BaseModel):
    """Incoming address payload."""
    address: str = Field(..., min_length=1, description="Raw address text to correct")


class AddressResponse(BaseModel):
    """Successful correction result."""
    corrected_address: str
    latitude: Any  # float | str ("NA")
    longitude: Any
    district: Optional[str] = None
    state: Optional[str] = None
    model_used: Optional[str] = None


class ErrorResponse(BaseModel):
    """Generic error wrapper."""
    error: str
    raw_output: Optional[str] = None
    script_stderr: Optional[str] = None


# ---------------------------------------------------------------------------
# Application lifespan – verify integrated.py exists at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook."""
    if not os.path.exists(INTEGRATED_SCRIPT_PATH):
        logger.critical("integrated.py not found at %s", INTEGRATED_SCRIPT_PATH)
        sys.exit(1)
    logger.info("Backend ready – integrated.py found at %s", INTEGRATED_SCRIPT_PATH)
    yield  # application runs
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Indian Address Correction API",
    version="1.0.0",
    description="Corrects noisy / abbreviated Indian addresses using T5 and LLaMA3 models.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper – run integrated.py as a subprocess
# ---------------------------------------------------------------------------
def _run_integrated(address: str) -> Dict[str, Any]:
    """
    Execute ``integrated.py`` with *address* as an argument and return
    the parsed JSON result.

    Raises ``HTTPException`` on timeouts, missing scripts, or parse errors.
    """
    try:
        result = subprocess.run(
            [sys.executable, INTEGRATED_SCRIPT_PATH, address],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=SUBPROCESS_TIMEOUT,
        )
    except FileNotFoundError:
        logger.error("integrated.py not found at %s", INTEGRATED_SCRIPT_PATH)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Backend script not found.",
        )
    except subprocess.TimeoutExpired:
        logger.warning("integrated.py timed out for input: %s", address)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Processing timed out.",
        )

    if result.stderr:
        logger.debug("stderr from integrated.py:\n%s", result.stderr.strip())

    # Extract the last JSON object from stdout (integrated.py may print
    # debug text before the final JSON blob).
    json_matches = re.findall(
        r"\{(?:[^{}]*|\{[^{}]*\})*\}", result.stdout, re.DOTALL
    )
    if json_matches:
        try:
            return json.loads(json_matches[-1])
        except json.JSONDecodeError:
            pass

    # Fallback – try parsing the entire stdout
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        logger.error(
            "Failed to parse JSON from integrated.py. Raw stdout:\n%s",
            result.stdout.strip(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse script output.",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post(
    "/process_address",
    response_model=AddressResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        504: {"model": ErrorResponse},
    },
    summary="Correct and geocode an Indian address",
)
async def process_address(payload: AddressRequest):
    """
    Accept a raw / noisy Indian address and return the corrected address
    along with latitude, longitude, district, and state.
    """
    logger.info("Received address: '%s'", payload.address)
    result = _run_integrated(payload.address)
    logger.info("Result: %s", result)
    return result


@app.get("/health", summary="Health check")
async def health():
    """Simple liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=True,
    )
