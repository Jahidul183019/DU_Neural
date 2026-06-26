"""QueueStorm – Evidence cross-referencing engine (pure rule-based, no LLM).

This module contains the core classification and evidence-matching logic
that drives ticket analysis.  Every function here is deterministic and
operates solely on the data passed in — no network calls, no LLM.

Public API
----------
- detect_language(complaint)              → "en" | "bn" | "mixed"
- find_relevant_transaction(complaint, h) → Optional[str]
- judge_evidence_verdict(complaint, h, id)→ "consistent" | "inconsistent" | "insufficient_data"
- classify_case(complaint, h, id, …)      → dict
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from models import TransactionEntry

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

# Bengali digit → ASCII digit translation table
_BN_DIGIT_TABLE = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _has_bengali(text: str) -> bool:
    """Return *True* if *text* contains at least one Bengali Unicode char."""
    return bool(re.search(r"[\u0980-\u09FF]", text))


def _has_latin(text: str) -> bool:
    """Return *True* if *text* contains at least one ASCII letter."""
    return bool(re.search(r"[a-zA-Z]", text))


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware UTC datetime."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _normalize_phone(raw: str) -> str:
    """Normalise a BD phone number to local 11-digit form ``01XXXXXXXXX``.

    Non-phone strings are returned unchanged so that merchant / agent IDs
    are not mangled.
    """
    if not raw or raw[0].isalpha():
        return raw
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 10:
        if digits.startswith("880") and len(digits) == 13:
            return "0" + digits[3:]
        # Only return digits if it looks like a phone number (e.g., 01...)
        if digits.startswith("01") and len(digits) == 11:
            return digits
    return raw  # not a phone number — return as-is


def _extract_amounts(text: str) -> List[float]:
    """Extract numeric monetary amounts, including Bengali digits.

    Matches patterns like ``5000``, ``5,000``, ``5000.50``, ``৫০০০``.
    """
    normalised = text.translate(_BN_DIGIT_TABLE)
    # Match digit groups optionally containing commas / decimal
    matches = re.findall(r"(?<!\w)[\d,]+(?:\.\d+)?(?!\w)", normalised)
    amounts: List[float] = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if val > 0:
                amounts.append(val)
        except ValueError:
            continue
    return amounts


def _extract_time_window(
    text: str,
    now: datetime,
) -> Optional[Tuple[datetime, datetime]]:
    """Derive a (start, end) time-range from natural-language keywords."""
    lower = text.lower()

    if "today" in lower:
        return (now.replace(hour=0, minute=0, second=0, microsecond=0), now)
        
    if any(kw in lower for kw in ("morning", "afternoon", "evening")):
        return (now - timedelta(hours=24), now)

    if "yesterday" in lower:
        start_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=24)
        return (start_of_yesterday, start_of_yesterday + timedelta(hours=24))

    # Explicit clock times like "2pm", "11 am"
    if re.search(r"\d{1,2}\s*(?:am|pm)", lower):
        return (now - timedelta(hours=24), now)

    return None


def _extract_counterparty_hints(text: str) -> List[str]:
    """Extract phone numbers, merchant IDs, and agent IDs from *text*."""
    hints: List[str] = []

    # Bangladesh mobile: +8801X…, 01X…
    phones = re.findall(r"(?:\+?880|0)1[3-9]\d{8}", text)
    hints.extend(phones)

    # Merchant IDs: MER-XXXX, MERCHANT-XXXX
    merchants = re.findall(r"(?:MER|MERCHANT)-\w+", text, re.IGNORECASE)
    hints.extend([m.upper() for m in merchants])

    # Agent IDs: AGENT-XXXX
    agents = re.findall(r"AGENT-\w+", text, re.IGNORECASE)
    hints.extend([a.upper() for a in agents])

    return hints


def _find_duplicate_pair(history: List[TransactionEntry]) -> Optional[str]:
    """Detect duplicate payments: same amount + counterparty ≤ 60 s apart.

    Returns the **second** transaction's ID, or ``None``.
    """
    if len(history) < 2:
        return None

    sorted_txns = sorted(history, key=lambda t: t.timestamp)
    for i in range(len(sorted_txns)):
        for j in range(i + 1, len(sorted_txns)):
            a, b = sorted_txns[i], sorted_txns[j]
            if a.amount == b.amount and a.counterparty == b.counterparty:
                try:
                    ta = _parse_timestamp(a.timestamp)
                    tb = _parse_timestamp(b.timestamp)
                    if abs((tb - ta).total_seconds()) <= 60:
                        return b.transaction_id
                except (ValueError, TypeError):
                    continue
    return None


def _lower_contains(text: str, keywords: List[str]) -> bool:
    """Return *True* if any *keyword* appears in the lowercased *text*."""
    lower = text.lower()
    return any(re.search(rf"(?:\b|\s|^){re.escape(kw)}(?:\b|\s|$)", lower) for kw in keywords)


def _counterparty_matches_hint(
    counterparty: str,
    hints: List[str],
) -> bool:
    """Check whether *counterparty* matches any extracted hint."""
    cp_norm = _normalize_phone(counterparty)
    for hint in hints:
        hint_norm = _normalize_phone(hint)
        if (
            hint in counterparty
            or hint_norm in cp_norm
            or counterparty in hint
            or cp_norm in hint_norm
        ):
            return True
    return False


def _get_relevant_amount(
    relevant_txn: Optional[TransactionEntry],
    complaint: str,
) -> float:
    """Best-effort extraction of the disputed amount."""
    if relevant_txn:
        return relevant_txn.amount
    amounts = _extract_amounts(complaint)
    return amounts[0] if len(amounts) == 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 1. detect_language
# ═══════════════════════════════════════════════════════════════════════════


def detect_language(complaint: str) -> str:
    """Detect complaint language → ``"en"`` | ``"bn"`` | ``"mixed"``.

    Bengali Unicode range: ``\\u0980``–``\\u09FF``.
    """
    has_bn = _has_bengali(complaint)
    has_lat = _has_latin(complaint)

    if has_bn and has_lat:
        return "mixed"
    if has_bn:
        return "bn"
    return "en"


# ═══════════════════════════════════════════════════════════════════════════
# 2. find_relevant_transaction
# ═══════════════════════════════════════════════════════════════════════════


def find_relevant_transaction(
    complaint: str,
    history: List[TransactionEntry],
) -> Optional[str]:
    """Return the ``transaction_id`` of the best-matching entry, or *None*.

    Matching priority:
      1. Amount extracted from complaint
      2. Time-proximity keywords (today, yesterday, …)
      3. Counterparty hints (phone numbers, merchant / agent IDs)

    Special case: duplicate-payment pairs (same amount + counterparty
    within 60 s) always return the **second** transaction's ID.
    """
    if not history:
        return None

    now = datetime.now(timezone.utc)
    try:
        latest = max(_parse_timestamp(t.timestamp) for t in history)
        if latest > now - timedelta(days=365):
            now = latest
    except (ValueError, TypeError):
        pass

    # ── Special case: duplicate payment ──────────────────────────────────
    if _lower_contains(complaint, _DUPLICATE_KW):
        dup_id = _find_duplicate_pair(history)
        if dup_id is not None:
            return dup_id

    # ── Step 1 – amounts ─────────────────────────────────────────────────
    amounts = _extract_amounts(complaint)

    # ── Step 2 – time window ─────────────────────────────────────────────
    time_window = _extract_time_window(complaint, now)

    # ── Step 3 – counterparty hints ──────────────────────────────────────
    cp_hints = _extract_counterparty_hints(complaint)

    # ── Candidate selection (progressive narrowing) ──────────────────────
    candidates = list(history)

    # Filter by amount
    if amounts:
        amount_match = [t for t in candidates if t.amount in amounts]
        if amount_match:
            candidates = amount_match

    # Filter by time proximity
    if time_window:
        start, end = time_window
        time_match: List[TransactionEntry] = []
        for t in candidates:
            try:
                ts = _parse_timestamp(t.timestamp)
                if start <= ts <= end:
                    time_match.append(t)
            except (ValueError, TypeError):
                continue
        if time_match:
            candidates = time_match

    # Filter by counterparty
    if cp_hints:
        cp_match = [
            t for t in candidates
            if _counterparty_matches_hint(t.counterparty, cp_hints)
        ]
        if cp_match:
            candidates = cp_match

    # ── Decision ─────────────────────────────────────────────────────────
    if len(candidates) == 1:
        return candidates[0].transaction_id
    # Zero or multiple → ambiguous, do NOT guess
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 3. judge_evidence_verdict
# ═══════════════════════════════════════════════════════════════════════════

_DUPLICATE_KW: List[str] = [
    "duplicate",
    "twice",
    "double",
    "two times",
    "charged twice",
    "double charge",
    "2 times",
]

_WRONG_TRANSFER_KW: List[str] = [
    "wrong transfer",
    "wrong number",
    "wrong person",
    "wrong recipient",
    "bhul number",
    "ভুল নাম্বার",
    "ভুল",
    "didn't get it",
    "did not get it",
]

_PAYMENT_FAILED_KW: List[str] = [
    "payment failed",
    "failed",
]

_CASH_IN_NOT_RECEIVED_KW: List[str] = [
    "didn't receive",
    "did not receive",
    "not received",
    "আসেনি",
    "দেখছি না",
]


def judge_evidence_verdict(
    complaint: str,
    history: List[TransactionEntry],
    relevant_id: Optional[str],
) -> str:
    """Return ``"consistent"`` | ``"inconsistent"`` | ``"insufficient_data"``.

    This is a *separate* judgment from ``find_relevant_transaction`` —
    it evaluates whether the transaction data **supports** or **contradicts**
    the customer's claim.
    """
    # ── No relevant transaction ──────────────────────────────────────────
    if relevant_id is None:
        return "insufficient_data"

    # Look up the relevant entry
    relevant_txn: Optional[TransactionEntry] = None
    for t in history:
        if t.transaction_id == relevant_id:
            relevant_txn = t
            break

    if relevant_txn is None:
        return "insufficient_data"
        
    now = datetime.now(timezone.utc)
    if history:
        try:
            latest = max(_parse_timestamp(t.timestamp) for t in history)
            if latest > now - timedelta(days=365):
                now = latest
        except (ValueError, TypeError):
            pass

    # ── INCONSISTENCY checks (run first — early return) ──────────────────

    # "wrong transfer" but established recipient (≥ 2 OTHER transfers to same cp)
    if _lower_contains(complaint, _WRONG_TRANSFER_KW):
        same_cp_count = sum(
            1 for t in history if t.counterparty == relevant_txn.counterparty and t.transaction_id != relevant_txn.transaction_id
        )
        if same_cp_count >= 2:
            return "inconsistent"

    # "payment failed" but transaction actually completed
    if _lower_contains(complaint, _PAYMENT_FAILED_KW):
        if relevant_txn.status == "completed":
            return "inconsistent"

    # "didn't receive cash in" but status is completed AND > 24 h ago
    if _lower_contains(complaint, _CASH_IN_NOT_RECEIVED_KW):
        if relevant_txn.status == "completed" and relevant_txn.type == "cash_in":
            try:
                txn_time = _parse_timestamp(relevant_txn.timestamp)
                if (now - txn_time).total_seconds() > 86_400:
                    return "inconsistent"
            except (ValueError, TypeError):
                pass
                
    # "duplicate payment" claimed but no duplicate pair exists
    if _lower_contains(complaint, _DUPLICATE_KW):
        if _find_duplicate_pair(history) is None:
            return "inconsistent"

    # ── CONSISTENCY checks ───────────────────────────────────────────────

    # "wrong transfer" and no other transfers to that counterparty
    if _lower_contains(complaint, _WRONG_TRANSFER_KW):
        same_cp_count = sum(
            1 for t in history if t.counterparty == relevant_txn.counterparty and t.transaction_id != relevant_txn.transaction_id
        )
        if same_cp_count == 0:
            return "consistent"

    # "payment failed" and status is indeed failed / pending
    if _lower_contains(complaint, _PAYMENT_FAILED_KW):
        if relevant_txn.status in ("failed", "pending"):
            return "consistent"

    # Duplicate payment pair exists in history
    if _lower_contains(complaint, _DUPLICATE_KW):
        if _find_duplicate_pair(history) is not None:
            return "consistent"

    # Settlement is pending and merchant complains of delay
    if _lower_contains(complaint, ["settlement", "not settled", "delay"]):
        if relevant_txn.type == "settlement" and relevant_txn.status == "pending":
            return "consistent"

    # Cash-in is pending and customer complains of non-receipt
    if _lower_contains(
        complaint, ["cash in", "cash-in", "not received", "not reflected", "ক্যাশ ইন", "আসেনি"]
    ):
        if relevant_txn.type == "cash_in" and relevant_txn.status == "pending":
            return "consistent"

    # Refund request logic (TKT-004)
    # If the user asks for a refund for a completed outgoing payment/transfer
    if _lower_contains(complaint, _REFUND_KW):
        if relevant_txn.status == "completed" and relevant_txn.type in ("payment", "transfer"):
            return "consistent"

    # ── Neither clear consistent nor inconsistent ────────────────────────
    return "insufficient_data"


# ═══════════════════════════════════════════════════════════════════════════
# 4. classify_case
# ═══════════════════════════════════════════════════════════════════════════

# ── Keyword banks (order mirrors the classification priority) ────────────

_PHISHING_KW: List[str] = [
    "otp",
    "pin",
    "password",
    "unknown call",
    "someone called",
    "hacker",
    "scam",
    "fraud call",
    "verify account",
]

_WRONG_XFER_KW: List[str] = [
    "wrong number",
    "wrong person",
    "wrong transfer",
    "wrong recipient",
    "bhul number",
    "ভুল নাম্বার",
    "ভুল",
    "didn't get it",
    "did not get it",
]

_PAY_FAILED_KW: List[str] = [
    "failed",
    "deducted",
    "not received",
    "balance cut",
    "payment failed",
    "কাটা গেছে",
]

_AGENT_KW: List[str] = [
    "agent",
    "cash in",
    "cash-in",
    "deposit",
    "not reflected",
    "balance nai",
    "এজেন্ট",
    "ক্যাশ ইন",
]

_MERCHANT_KW: List[str] = [
    "settlement",
    "merchant",
    "not settled",
    "sales",
]

_REFUND_KW: List[str] = [
    "refund",
    "money back",
    "return my money",
    "ফেরত",
]


def classify_case(
    complaint: str,
    history: List[TransactionEntry],
    relevant_id: Optional[str],
    user_type: Optional[str],
    *,
    channel: Optional[str] = None,
    evidence_verdict: Optional[str] = None,
) -> Dict:
    """Classify a ticket into case-type, severity, department, etc.

    Classification rules are checked **in priority order** (phishing first).
    After initial classification, ``human_review_required`` overrides are
    applied.

    Returns
    -------
    dict
        Keys: ``case_type``, ``severity``, ``department``,
        ``human_review_required``, ``reason_codes``.
    """
    # ── Resolve the relevant transaction ─────────────────────────────────
    relevant_txn: Optional[TransactionEntry] = None
    if relevant_id:
        for t in history:
            if t.transaction_id == relevant_id:
                relevant_txn = t
                break

    result: Optional[Dict] = None

    # ─────────────────────────────────────────────────────────────────────
    # Rule 1 — Phishing / Social Engineering
    # ─────────────────────────────────────────────────────────────────────
    if result is None and _lower_contains(complaint, _PHISHING_KW):
        result = {
            "case_type": "phishing_or_social_engineering",
            "severity": "critical",
            "department": "fraud_risk",
            "human_review_required": True,
            "reason_codes": ["phishing_detected", "fraud_risk"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 2 — Duplicate Payment
    # ─────────────────────────────────────────────────────────────────────
    if result is None and _lower_contains(complaint, _DUPLICATE_KW):
        has_duplicate = _find_duplicate_pair(history) is not None
        result = {
            "case_type": "duplicate_payment",
            "severity": "high",
            "department": "payments_ops",
            "human_review_required": True,
            "reason_codes": ["duplicate_payment_detected" if has_duplicate else "duplicate_payment_claimed"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 3 — Wrong Transfer
    # ─────────────────────────────────────────────────────────────────────
    if result is None and _lower_contains(complaint, _WRONG_XFER_KW):
        amount = _get_relevant_amount(relevant_txn, complaint)
        result = {
            "case_type": "wrong_transfer",
            "severity": "high" if amount > 2000 else "medium",
            "department": "dispute_resolution",
            "human_review_required": True,
            "reason_codes": ["wrong_transfer_claimed"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 4 — Payment Failed
    # ─────────────────────────────────────────────────────────────────────
    if result is None and _lower_contains(complaint, _PAY_FAILED_KW):
        amount = _get_relevant_amount(relevant_txn, complaint)
        severity = "high" if amount > 2000 else ("medium" if amount > 500 else "low")
        result = {
            "case_type": "payment_failed",
            "severity": severity,
            "department": "payments_ops",
            "human_review_required": False,
            "reason_codes": ["payment_failure_reported"],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 5 — Agent Cash-In Issue
    # ─────────────────────────────────────────────────────────────────────
    if result is None:
        agent_keyword_hit = _lower_contains(complaint, _AGENT_KW)
        agent_user = user_type == "agent"
        agent_counterparty = (
            relevant_txn is not None
            and relevant_txn.counterparty.upper().startswith("AGENT-")
        )
        if agent_keyword_hit or agent_user or agent_counterparty:
            result = {
                "case_type": "agent_cash_in_issue",
                "severity": "high",
                "department": "agent_operations",
                "human_review_required": True,
                "reason_codes": ["agent_cash_in_reported"],
            }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 6 — Merchant Settlement Delay
    # ─────────────────────────────────────────────────────────────────────
    if result is None:
        merchant_keyword_hit = _lower_contains(complaint, _MERCHANT_KW)
        merchant_user = user_type == "merchant"
        merchant_channel = channel == "merchant_portal"
        if user_type != "customer" and (merchant_keyword_hit or merchant_user or merchant_channel):
            result = {
                "case_type": "merchant_settlement_delay",
                "severity": "medium",
                "department": "merchant_operations",
                "human_review_required": False,
                "reason_codes": ["merchant_settlement_issue"],
            }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 7 — Refund Request
    # ─────────────────────────────────────────────────────────────────────
    if result is None and _lower_contains(complaint, _REFUND_KW):
        amount = _get_relevant_amount(relevant_txn, complaint)
        if evidence_verdict == "inconsistent" or amount > 5000:
            result = {
                "case_type": "refund_request",
                "severity": "medium",
                "department": "dispute_resolution",
                "human_review_required": True,
                "reason_codes": ["refund_request_escalated"],
            }
        else:
            result = {
                "case_type": "refund_request",
                "severity": "low",
                "department": "customer_support",
                "human_review_required": False,
                "reason_codes": ["refund_request_standard"],
            }

    # ─────────────────────────────────────────────────────────────────────
    # Rule 8 — Default / Other
    # ─────────────────────────────────────────────────────────────────────
    if result is None:
        result = {
            "case_type": "other",
            "severity": "low",
            "department": "customer_support",
            "human_review_required": False,
            "reason_codes": ["unclassified"],
        }

    # ═════════════════════════════════════════════════════════════════════
    # human_review_required overrides (applied AFTER classification)
    # ═════════════════════════════════════════════════════════════════════
    max_amount = _get_relevant_amount(relevant_txn, complaint)

    # ── "Always True" conditions ─────────────────────────────────────────
    always_true = False

    if evidence_verdict == "inconsistent":
        always_true = True
        if "evidence_inconsistent" not in result["reason_codes"]:
            result["reason_codes"].append("evidence_inconsistent")

    if max_amount > 5000 and user_type == "customer":
        always_true = True
        if "high_value_transaction" not in result["reason_codes"]:
            result["reason_codes"].append("high_value_transaction")

    if result["case_type"] == "phishing_or_social_engineering":
        always_true = True
    elif result["case_type"] in ("wrong_transfer", "duplicate_payment"):
        if evidence_verdict != "insufficient_data":
            always_true = True

    if always_true:
        result["human_review_required"] = True

    if evidence_verdict == "insufficient_data" and result["case_type"] != "phishing_or_social_engineering":
        result["human_review_required"] = False

    # ── "Always False" condition (only when no "Always True" fired) ──────
    #    Applies when the only viable next step is to ask the customer
    #    a clarifying question (ambiguous match, no relevant txn,
    #    insufficient evidence, low severity).
    if not always_true:
        if (
            relevant_id is None
            and evidence_verdict == "insufficient_data"
            and result["severity"] == "low"
        ):
            result["human_review_required"] = False

    return result
