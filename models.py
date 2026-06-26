"""Pydantic v2 data models for QueueStorm ticket analysis."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TransactionEntry(BaseModel):
    """A single transaction record from the user's history."""

    transaction_id: str
    timestamp: str
    type: Literal[
        "transfer",
        "payment",
        "cash_in",
        "cash_out",
        "settlement",
        "refund",
    ]
    amount: float
    counterparty: str
    status: Literal["completed", "failed", "pending", "reversed"]


class TicketRequest(BaseModel):
    """Incoming support-ticket payload sent by the client."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    complaint: str = Field(..., description="Raw complaint text from the user")
    language: Optional[Literal["en", "bn", "mixed"]] = None
    channel: Optional[
        Literal[
            "in_app_chat",
            "call_center",
            "email",
            "merchant_portal",
            "field_agent",
        ]
    ] = None
    user_type: Optional[Literal["customer", "merchant", "agent", "unknown"]] = None
    campaign_context: Optional[str] = None
    transaction_history: Optional[List[TransactionEntry]] = []
    metadata: Optional[dict] = None


class TicketResponse(BaseModel):
    """Structured analysis result returned to the client."""

    ticket_id: str
    relevant_transaction_id: Optional[str]  # Required field, but can be null
    evidence_verdict: Literal["consistent", "inconsistent", "insufficient_data"]
    case_type: Literal[
        "wrong_transfer",
        "payment_failed",
        "refund_request",
        "duplicate_payment",
        "merchant_settlement_delay",
        "agent_cash_in_issue",
        "phishing_or_social_engineering",
        "other",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    department: Literal[
        "customer_support",
        "dispute_resolution",
        "payments_ops",
        "merchant_operations",
        "agent_operations",
        "fraud_risk",
    ]
    agent_summary: str
    recommended_next_action: str
    customer_reply: str
    human_review_required: bool
    confidence: Optional[float] = None
    reason_codes: Optional[List[str]] = None
