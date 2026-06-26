"""QueueStorm – Ticket analysis pipeline.

Orchestrates the full analysis flow:
  1. Language detection
  2. Prompt-injection pre-screen
  3. Evidence extraction  (rule-based)
  4. Classification        (rule-based)
  5. LLM text generation   (Gemini API via httpx)
  6. Response parsing
  7. Response assembly
  8. Safety post-filter     (unconditional)
  9. Return
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from fastapi import HTTPException

from evidence import (
    classify_case,
    detect_language,
    find_relevant_transaction,
    judge_evidence_verdict,
)
from models import TicketRequest, TicketResponse
from prompts import SYSTEM_PROMPT, build_user_prompt
from safety import _RE_PROMPT_INJECTION, post_process_safety

logger = logging.getLogger("queuestorm.analyzer")

# ═══════════════════════════════════════════════════════════════════════════
# Fallback text (used when LLM call fails or is skipped)
# ═══════════════════════════════════════════════════════════════════════════

_FALLBACK_SUMMARY = "Automated analysis completed. Manual review recommended."
_FALLBACK_NEXT_ACTION = "Route to human agent for review."

_FALLBACK_REPLY_EN = (
    "Thank you for contacting support. A team member will review your "
    "case and respond through official channels. Please do not share "
    "your PIN or OTP with anyone."
)

_FALLBACK_REPLY_BN = (
    "আপনার অভিযোগের জন্য ধন্যবাদ। আমাদের দল আপনার সাথে অফিসিয়াল "
    "চ্যানেলে যোগাযোগ করবে। আপনার পিন বা ওটিপি কারো সাথে শেয়ার "
    "করবেন না।"
)

# Strip ```json … ``` fences that LLMs sometimes add
_RE_MD_FENCES = re.compile(r"^```(?:json)?\s*\n?|```\s*$", re.MULTILINE)

# Gemini API config
_GEMINI_MODEL = "gemini-2.0-flash"
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{_GEMINI_MODEL}:generateContent"
)
_LLM_TIMEOUT = 25.0  # seconds


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fallback_reply(language: str) -> str:
    """Pick the right fallback customer_reply by language."""
    if language == "bn":
        return _FALLBACK_REPLY_BN
    return _FALLBACK_REPLY_EN


async def _call_gemini(system: str, user_prompt: str, api_key: str) -> dict:
    """POST to Gemini ``generateContent`` and return parsed JSON body.

    Raises on timeout, HTTP errors, or missing fields so the caller can
    fall back gracefully.
    """
    async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
        resp = await client.post(
            _GEMINI_ENDPOINT,
            params={"key": api_key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": [
                    {"role": "user", "parts": [{"text": user_prompt}]},
                ],
                "generationConfig": {
                    "maxOutputTokens": 1000,
                    "temperature": 0.3,
                },
            },
        )
        resp.raise_for_status()
        return resp.json()


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


async def analyze_ticket(ticket: TicketRequest) -> TicketResponse:
    """Run the full 9-step analysis pipeline and return a safe response."""
    try:
        # ─────────────────────────────────────────────────────────────────
        # Step 1 — Language detection
        # ─────────────────────────────────────────────────────────────────
        detected_lang = detect_language(ticket.complaint)
        effective_language = ticket.language or detected_lang

        # ─────────────────────────────────────────────────────────────────
        # Step 2 — Prompt-injection pre-screen (flag only)
        # ─────────────────────────────────────────────────────────────────
        injection_detected = bool(_RE_PROMPT_INJECTION.search(ticket.complaint))
        if injection_detected:
            logger.warning(
                "Prompt-injection attempt pre-screened in ticket %s",
                ticket.ticket_id,
            )

        # ─────────────────────────────────────────────────────────────────
        # Step 3 — Evidence extraction (rule-based, NO AI)
        # ─────────────────────────────────────────────────────────────────
        history = ticket.transaction_history or []
        relevant_id = find_relevant_transaction(ticket.complaint, history)
        verdict = judge_evidence_verdict(ticket.complaint, history, relevant_id)

        # ─────────────────────────────────────────────────────────────────
        # Step 4 — Classification (rule-based, NO AI)
        # ─────────────────────────────────────────────────────────────────
        classification = classify_case(
            ticket.complaint,
            history,
            relevant_id,
            ticket.user_type,
            channel=ticket.channel,
            evidence_verdict=verdict,
        )

        # ─────────────────────────────────────────────────────────────────
        # Step 5 + 6 — LLM call & parse
        # ─────────────────────────────────────────────────────────────────
        used_fallback = False
        agent_summary = _FALLBACK_SUMMARY
        recommended_next_action = _FALLBACK_NEXT_ACTION
        customer_reply = _fallback_reply(effective_language)

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("GEMINI_API_KEY not set — using fallback responses")
            used_fallback = True
        else:
            try:
                user_prompt = build_user_prompt(
                    ticket,
                    classification,
                    relevant_id,
                    verdict,
                    language=effective_language,
                )

                body = await _call_gemini(SYSTEM_PROMPT, user_prompt, api_key)

                # Extract text from Gemini response
                raw_text = (
                    body["candidates"][0]["content"]["parts"][0]["text"]
                )

                # Strip markdown fences if present
                clean = _RE_MD_FENCES.sub("", raw_text).strip()
                parsed = json.loads(clean)

                # Pull the three fields (fall back individually)
                agent_summary = parsed.get("agent_summary", agent_summary)
                recommended_next_action = parsed.get(
                    "recommended_next_action", recommended_next_action
                )
                customer_reply = parsed.get("customer_reply", customer_reply)

            except (
                httpx.TimeoutException,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
                KeyError,
                IndexError,
                TypeError,
            ) as exc:
                logger.warning(
                    "LLM call failed for ticket %s: %s: %s",
                    ticket.ticket_id,
                    type(exc).__name__,
                    exc,
                )
                used_fallback = True

        # ─────────────────────────────────────────────────────────────────
        # Step 7 — Assemble TicketResponse
        # ─────────────────────────────────────────────────────────────────

        # Confidence scoring
        if used_fallback:
            confidence = 0.50
        elif verdict == "consistent" and relevant_id is not None:
            confidence = 0.90
        elif verdict == "inconsistent":
            confidence = 0.80
        else:  # insufficient_data or other
            confidence = 0.65

        # Reason codes (from classification + injection flag)
        reason_codes: list[str] = list(
            classification.get("reason_codes", [])
        )
        if injection_detected and "prompt_injection_attempt_detected" not in reason_codes:
            reason_codes.append("prompt_injection_attempt_detected")

        response = TicketResponse(
            ticket_id=ticket.ticket_id,
            relevant_transaction_id=relevant_id,
            evidence_verdict=verdict,
            case_type=classification["case_type"],
            severity=classification["severity"],
            department=classification["department"],
            agent_summary=agent_summary,
            recommended_next_action=recommended_next_action,
            customer_reply=customer_reply,
            human_review_required=classification["human_review_required"],
            confidence=confidence,
            reason_codes=reason_codes,
        )

        # ─────────────────────────────────────────────────────────────────
        # Step 8 — Safety post-filter (UNCONDITIONAL)
        # ─────────────────────────────────────────────────────────────────
        response = post_process_safety(response, ticket)

        # ─────────────────────────────────────────────────────────────────
        # Step 9 — Return
        # ─────────────────────────────────────────────────────────────────
        return response

    except HTTPException:
        # Re-raise FastAPI exceptions as-is (e.g. our own 503)
        raise
    except Exception:
        logger.exception(
            "Unhandled error in analyze_ticket for %s", ticket.ticket_id
        )
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable",
        )
