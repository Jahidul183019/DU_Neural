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

from utils import detect_language
from matcher import find_relevant_transaction
from verdict import judge_evidence_verdict
from classifier import classify_case
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

# Groq API config
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
_LLM_TIMEOUT = 25.0  # seconds


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _fallback_reply(language: str) -> str:
    """Pick the right fallback customer_reply by language."""
    if language == "bn":
        return _FALLBACK_REPLY_BN
    return _FALLBACK_REPLY_EN


import asyncio

async def _call_groq(system: str, user_prompt: str, api_key: str, max_retries: int = 2) -> str:
    """POST to Groq chat completions and return the generated text.

    Includes automatic exponential backoff for HTTP 429 (Rate Limit) and 50x errors.
    """
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=_LLM_TIMEOUT) as client:
                resp = await client.post(
                    _GROQ_ENDPOINT,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": _GROQ_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 1000,
                        "response_format": {"type": "json_object"},
                    },
                )
                resp.raise_for_status()
                body = resp.json()
                return body["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Groq API returned %d. Retrying in %d seconds (attempt %d/%d)...",
                        status, wait_time, attempt + 1, max_retries
                    )
                    await asyncio.sleep(wait_time)
                    continue
            raise
        except httpx.TimeoutException:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                logger.warning("Groq API timeout. Retrying in %d seconds...", wait_time)
                await asyncio.sleep(wait_time)
                continue
            raise


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

        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            logger.warning("GROQ_API_KEY not set — using fallback responses")
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

                raw_text = await _call_groq(SYSTEM_PROMPT, user_prompt, api_key)

                # Strip markdown fences if present
                clean = _RE_MD_FENCES.sub("", raw_text).strip()
                parsed = json.loads(clean)

                # Pull the three fields (fall back individually, handle explicit nulls)
                new_summary = parsed.get("agent_summary")
                new_action = parsed.get("recommended_next_action")
                new_reply = parsed.get("customer_reply")

                if not new_summary or not new_action or not new_reply:
                    used_fallback = True

                agent_summary = new_summary or agent_summary
                recommended_next_action = new_action or recommended_next_action
                customer_reply = new_reply or customer_reply

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
        response = post_process_safety(response, ticket, effective_language)
        
        if response.reason_codes and any(code.startswith("safety_") for code in response.reason_codes):
            if response.confidence is not None:
                response.confidence = min(response.confidence, 0.50)

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
