from datetime import datetime, timedelta, timezone
from typing import List, Optional

from models import TransactionEntry
from utils import (
    _parse_timestamp,
    _extract_amounts,
    _lower_contains,
)
from matcher import _find_duplicate_pair, _DUPLICATE_KW

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
    "mistakenly sent",
    "accidentally sent",
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
    if relevant_id is None:
        return "insufficient_data"

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

    complaint_amounts = _extract_amounts(complaint)
    if complaint_amounts and relevant_txn.amount is not None:
        if not any(
            abs(relevant_txn.amount - a) < 0.01 for a in complaint_amounts
        ):
            return "inconsistent"

    _DEDUCTED_TERMS = ["deducted", "deduct", "cut", "taken", "charged",
                       "money gone", "balance gone", "কেটে", "কেটেছে", "কাটা"]
    if relevant_txn.status == "reversed" and _lower_contains(complaint, _DEDUCTED_TERMS):
        return "inconsistent"

    if _lower_contains(complaint, _WRONG_TRANSFER_KW):
        same_cp_count = sum(
            1 for t in history if t.counterparty == relevant_txn.counterparty and t.transaction_id != relevant_txn.transaction_id
        )
        if same_cp_count >= 2:
            return "inconsistent"

    if _lower_contains(complaint, _PAYMENT_FAILED_KW):
        if relevant_txn.status == "completed":
            return "inconsistent"

    if _lower_contains(complaint, _CASH_IN_NOT_RECEIVED_KW):
        if relevant_txn.status == "completed" and relevant_txn.type == "cash_in":
            try:
                txn_time = _parse_timestamp(relevant_txn.timestamp)
                if (now - txn_time).total_seconds() > 86_400:
                    return "inconsistent"
            except (ValueError, TypeError):
                pass
                
    if _lower_contains(complaint, _DUPLICATE_KW):
        if _find_duplicate_pair(history) is None:
            return "inconsistent"

    if _lower_contains(complaint, _WRONG_TRANSFER_KW):
        same_cp_count = sum(
            1 for t in history if t.counterparty == relevant_txn.counterparty and t.transaction_id != relevant_txn.transaction_id
        )
        if same_cp_count == 0:
            return "consistent"

    if _lower_contains(complaint, _PAYMENT_FAILED_KW):
        if relevant_txn.status in ("failed", "pending"):
            return "consistent"

    if _lower_contains(complaint, _DUPLICATE_KW):
        if _find_duplicate_pair(history) is not None:
            return "consistent"

    if _lower_contains(complaint, ["settlement", "not settled", "delay"]):
        if relevant_txn.type == "settlement" and relevant_txn.status == "pending":
            return "consistent"

    if _lower_contains(
        complaint, ["cash in", "cash-in", "not received", "not reflected", "ক্যাশ ইন", "আসেনি"]
    ):
        if relevant_txn.type == "cash_in" and relevant_txn.status == "pending":
            return "consistent"

    _REFUND_KW = ["refund", "money back", "return my money", "ফেরত"]
    if _lower_contains(complaint, _REFUND_KW):
        if relevant_txn.status == "completed" and relevant_txn.type in ("payment", "transfer"):
            return "consistent"

    return "insufficient_data"
