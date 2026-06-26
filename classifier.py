from typing import Dict, List, Optional
from models import TransactionEntry

from utils import _lower_contains, _get_relevant_amount
from matcher import _find_duplicate_pair, _DUPLICATE_KW
from verdict import (
    _WRONG_TRANSFER_KW,
    _PAYMENT_FAILED_KW,
)

_PHISHING_KW: List[str] = [
    "otp", "pin", "password", "passcode", "cvv", "card number",
    "unknown call", "someone called", "hacker", "scam",
    "fraud call", "fraud", "verify account", "suspicious",
    "lottery", "prize", "\u0993\u099f\u09bf\u09aa\u09bf",
    "\u09aa\u09bf\u09a8", "\u09aa\u09be\u09b8\u0993\u09af\u09bc\u09be\u09b0\u09cd\u09a1",
    "\u09aa\u09cd\u09b0\u09a4\u09be\u09b0\u0995",
]

_IMPERSONATION_KW: List[str] = [
    "from bkash", "bkash called", "agent called", "company called",
    "from bank", "called me", "call from",
    "\u09ac\u09bf\u0995\u09be\u09b6 \u09a5\u09c7\u0995\u09c7",
    "\u09ab\u09cb\u09a8 \u09a6\u09bf\u09af\u09bc\u09c7",
]

_THREAT_KW: List[str] = [
    "account will be blocked", "account blocked", "account block",
    "account freeze", "blocked if", "block if", "kyc update",
    "verify kyc", "update kyc",
    "\u0985\u09cd\u09af\u09be\u0995\u09be\u0989\u09a8\u09cd\u099f \u09ac\u09cd\u09b2\u0995",
    "\u09ac\u09cd\u09b2\u0995",
]

_AGENT_KW: List[str] = [
    "agent", "cash in", "cash-in", "deposit", "not reflected",
    "balance nai", "এজেন্ট", "ক্যাশ ইন",
]

_MERCHANT_KW: List[str] = [
    "settlement", "merchant", "not settled", "sales",
]

_REFUND_KW: List[str] = [
    "refund", "money back", "return my money", "ফেরত",
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
    relevant_txn: Optional[TransactionEntry] = None
    if relevant_id:
        for t in history:
            if t.transaction_id == relevant_id:
                relevant_txn = t
                break

    result: Optional[Dict] = None

    if result is None:
        is_phishing = _lower_contains(complaint, _PHISHING_KW)
        is_impersonation_threat = (
            _lower_contains(complaint, _IMPERSONATION_KW)
            and _lower_contains(complaint, _THREAT_KW)
        )
        if is_phishing or is_impersonation_threat:
            result = {
                "case_type": "phishing_or_social_engineering",
                "severity": "critical",
                "department": "fraud_risk",
                "human_review_required": True,
                "reason_codes": ["phishing_detected", "fraud_risk"],
            }

    if result is None and _lower_contains(complaint, _DUPLICATE_KW):
        has_duplicate = _find_duplicate_pair(history) is not None
        result = {
            "case_type": "duplicate_payment",
            "severity": "high",
            "department": "payments_ops",
            "human_review_required": True,
            "reason_codes": ["duplicate_payment_detected" if has_duplicate else "duplicate_payment_claimed"],
        }

    if result is None and _lower_contains(complaint, _WRONG_TRANSFER_KW):
        amount = _get_relevant_amount(relevant_txn, complaint)
        result = {
            "case_type": "wrong_transfer",
            "severity": "high" if amount > 2000 else "medium",
            "department": "dispute_resolution",
            "human_review_required": True,
            "reason_codes": ["wrong_transfer_claimed"],
        }

    if result is None and _lower_contains(complaint, _PAYMENT_FAILED_KW):
        amount = _get_relevant_amount(relevant_txn, complaint)
        severity = "high" if amount > 2000 else ("medium" if amount > 500 else "low")
        result = {
            "case_type": "payment_failed",
            "severity": severity,
            "department": "payments_ops",
            "human_review_required": False,
            "reason_codes": ["payment_failure_reported"],
        }

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

    if result is None:
        result = {
            "case_type": "other",
            "severity": "low",
            "department": "customer_support",
            "human_review_required": False,
            "reason_codes": ["unclassified"],
        }

    max_amount = _get_relevant_amount(relevant_txn, complaint)
    _SEVERITY_ORDER = ["low", "medium", "high", "critical"]

    if max_amount >= 50000 and result["severity"] != "critical":
        cur_idx = _SEVERITY_ORDER.index(result["severity"])
        result["severity"] = _SEVERITY_ORDER[min(cur_idx + 1, len(_SEVERITY_ORDER) - 1)]
        if "high_value_transaction" not in result["reason_codes"]:
            result["reason_codes"].append("high_value_transaction")

    always_true = False

    if evidence_verdict == "inconsistent":
        always_true = True
        if "evidence_inconsistent" not in result["reason_codes"]:
            result["reason_codes"].append("evidence_inconsistent")

    if max_amount >= 50000:
        always_true = True
        if "high_value_transaction" not in result["reason_codes"]:
            result["reason_codes"].append("high_value_transaction")

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

    if evidence_verdict == "insufficient_data" and result["case_type"] != "phishing_or_social_engineering" and not always_true:
        result["human_review_required"] = False

    if not always_true:
        if (
            relevant_id is None
            and evidence_verdict == "insufficient_data"
            and result["severity"] == "low"
        ):
            result["human_review_required"] = False

    return result
