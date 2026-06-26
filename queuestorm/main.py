"""QueueStorm – FastAPI entry-point."""

import time
import logging
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from models import TicketRequest, TicketResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("queuestorm")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="QueueStorm",
    description="AI-powered support-ticket analysis API",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Middleware – request logging
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "%s | %s %s | %s | %.2f ms",
        datetime.now(timezone.utc).isoformat(),
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return 422 for Pydantic / FastAPI validation errors."""
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Surface known HTTP errors (400, 404, …) without leaking internals."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all: return 500 — NEVER expose stack traces or secrets."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze-ticket", response_model=TicketResponse)
async def analyze_ticket(ticket: TicketRequest):
    # Reject empty / whitespace-only complaints with 422
    if not ticket.complaint or not ticket.complaint.strip():
        return JSONResponse(
            status_code=422,
            content={"detail": "complaint must not be empty or whitespace-only"},
        )

    # --- Hardcoded placeholder response (analysis not yet wired) ----------
    return TicketResponse(
        ticket_id=ticket.ticket_id,
        relevant_transaction_id=None,
        evidence_verdict="insufficient_data",
        case_type="other",
        severity="medium",
        department="customer_support",
        agent_summary="Placeholder: analysis not yet implemented.",
        recommended_next_action="Route to human agent for review.",
        customer_reply=(
            "Thank you for contacting support. A team member will review "
            "your case and respond through official channels. Please do not "
            "share your PIN or OTP with anyone."
        ),
        human_review_required=True,
        confidence=0.5,
        reason_codes=["placeholder"],
    )
