"""QueueStorm – Safety guardrails (hard rule-based post-processor).

This module is the **last line of defense** before a ``TicketResponse`` is
returned to the caller.  Every rule here is deterministic, regex-driven,
and runs on *every* response — regardless of how it was produced.

Public API
----------
- post_process_safety(response, ticket) → TicketResponse
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import List

from models import TicketRequest, TicketResponse

logger = logging.getLogger("queuestorm.safety")

# ═══════════════════════════════════════════════════════════════════════════
# Compiled regex patterns (case-insensitive)
# ═══════════════════════════════════════════════════════════════════════════

# Rule 1 — credential solicitation
_RE_CREDENTIAL = re.compile(
    r"\b(share|send|provide|give|enter|type|tell|submit|confirm|what is|need|require)"
    r".{0,40}"
    r"(pin|otp|one.time|password|passcode|card.number|full.card)\b",
    re.IGNORECASE,
)

# Rule 2 — unauthorized refund / reversal / unblock promise
_RE_REFUND_PROMISE = re.compile(
    r"\b(we will|we'll|you will|you'll|i will|going to)"
    r".{0,30}"
    r"(refund|reverse|credit|return your|unblock|recover)\b",
    re.IGNORECASE,
)

# Rule 3 — third-party redirect
_RE_THIRD_PARTY = re.compile(
    r"\b(contact|call|visit|reach out to|message)"
    r".{0,40}"
    r"(third.party|external|another (service|provider|number|agent)|whatsapp|facebook)\b",
    re.IGNORECASE,
)

# Rule 4 — phishing: attempt to verify the caller
_RE_VERIFY_CALLER = re.compile(
    r"\b(we will check|we will verify|we will investigate)"
    r".{0,30}"
    r"(call|caller|person|number)\b",
    re.IGNORECASE,
)

# Rule 6 — prompt injection patterns
_RE_PROMPT_INJECTION = re.compile(
    r"(ignore (previous|all|prior|your) instructions?"
    r"|disregard"
    r"|forget your (rules|instructions)"
    r"|you are now"
    r"|pretend you are"
    r"|act as if"
    r"|return.*refund.*approved"
    r"|confirm.*refund)",
    re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Safe fallback messages
# ═══════════════════════════════════════════════════════════════════════════

_SAFE_REPLY_EN = (
    "We have noted your concern. Our team will review your case and "
    "contact you through official support channels. Please do not share "
    "your PIN, OTP, or password with anyone, including support agents."
)

_SAFE_REPLY_BN = (
    "আমরা আপনার অভিযোগ নোট করেছি। আমাদের দল অফিসিয়াল চ্যানেলের মাধ্যমে "
    "আপনার সাথে যোগাযোগ করবে। অনুগ্রহ করে কারো সাথে আপনার পিন, ওটিপি বা "
    "পাসওয়ার্ড শেয়ার করবেন না।"
)

_SAFE_REFUND_CLAUSE = (
    "any eligible amount will be returned through official channels"
)

_SAFE_THIRD_PARTY_REPLY = (
    "Please contact our official support through the app or call our "
    "helpline. Do not share your credentials with anyone."
)

_SAFE_PHISHING_REPLY = (
    "We are aware of fraudulent activity in circulation. Please never "
    "share your PIN, OTP, or password with anyone. Our fraud team has "
    "been notified and will take appropriate action."
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _ensure_reason_codes(codes: List[str] | None) -> List[str]:
    """Guarantee a mutable list for reason_codes."""
    return list(codes) if codes else []


def _add_reason(codes: List[str], reason: str) -> List[str]:
    """Append *reason* to *codes* if not already present."""
    if reason not in codes:
        codes.append(reason)
    return codes


def _pick_language_reply(language: str | None) -> str:
    """Return the safe credential-violation fallback in the right language."""
    if language == "bn":
        return _SAFE_REPLY_BN
    # "en", "mixed", or None → English
    return _SAFE_REPLY_EN


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def post_process_safety(
    response: TicketResponse,
    ticket: TicketRequest,
) -> TicketResponse:
    """Apply hard safety rules to *response* before it leaves the system.

    This function **never raises**.  It returns a corrected copy of the
    response — the original is not mutated.

    Rules executed (in order):
      6. Prompt-injection guard  (flag only, no field mutation)
      5. Hard-enforce ticket_id echo
      1. Never ask for credentials
      2. Never confirm unauthorized refund / reversal / unblock
      3. Never redirect to third party
      4. Phishing-case lockdown
    """
    # Work on a mutable copy so callers keep their original
    safe = response.model_copy(deep=True)
    codes = _ensure_reason_codes(safe.reason_codes)

    # ─────────────────────────────────────────────────────────────────────
    # Rule 6 — Prompt-injection guard (runs FIRST)
    # ─────────────────────────────────────────────────────────────────────
    if _RE_PROMPT_INJECTION.search(ticket.complaint):
        logger.warning(
            "Prompt-injection attempt detected in ticket %s",
            ticket.ticket_id,
        )
        codes = _add_reason(codes, "prompt_injection_attempt_detected")

    # ─────────────────────────────────────────────────────────────────────
    # Rule 5 — Hard-enforce ticket_id echo
    # ─────────────────────────────────────────────────────────────────────
    safe.ticket_id = ticket.ticket_id

    # ─────────────────────────────────────────────────────────────────────
    # Rule 1 — Never ask for credentials (PIN / OTP / password / card)
    # ─────────────────────────────────────────────────────────────────────
    if _RE_CREDENTIAL.search(safe.customer_reply):
        safe.customer_reply = _pick_language_reply(ticket.language)
        codes = _add_reason(codes, "safety_credential_violation_corrected")

    # ─────────────────────────────────────────────────────────────────────
    # Rule 2 — Never confirm unauthorized refund / reversal / unblock
    # ─────────────────────────────────────────────────────────────────────
    # Check both customer_reply and recommended_next_action
    if _RE_REFUND_PROMISE.search(safe.customer_reply):
        safe.customer_reply = _RE_REFUND_PROMISE.sub(
            _SAFE_REFUND_CLAUSE,
            safe.customer_reply,
        )
        codes = _add_reason(codes, "safety_refund_language_corrected")

    if _RE_REFUND_PROMISE.search(safe.recommended_next_action):
        safe.recommended_next_action = _RE_REFUND_PROMISE.sub(
            _SAFE_REFUND_CLAUSE,
            safe.recommended_next_action,
        )
        codes = _add_reason(codes, "safety_refund_language_corrected")

    # ─────────────────────────────────────────────────────────────────────
    # Rule 3 — Never redirect to third party
    # ─────────────────────────────────────────────────────────────────────
    if _RE_THIRD_PARTY.search(safe.customer_reply):
        safe.customer_reply = _SAFE_THIRD_PARTY_REPLY
        codes = _add_reason(codes, "safety_third_party_redirect_corrected")

    # ─────────────────────────────────────────────────────────────────────
    # Rule 4 — Phishing cases: lock down fields + scan for verify attempts
    # ─────────────────────────────────────────────────────────────────────
    if safe.case_type == "phishing_or_social_engineering":
        safe.severity = "critical"
        safe.department = "fraud_risk"
        safe.human_review_required = True

        if _RE_VERIFY_CALLER.search(safe.customer_reply):
            safe.customer_reply = _SAFE_PHISHING_REPLY
            codes = _add_reason(codes, "safety_phishing_verify_corrected")

    # ─────────────────────────────────────────────────────────────────────
    # Commit reason_codes back
    # ─────────────────────────────────────────────────────────────────────
    safe.reason_codes = codes

    return safe
