from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from models import TransactionEntry
from utils import (
    _parse_timestamp,
    _extract_amounts,
    _extract_time_window,
    _extract_counterparty_hints,
    _lower_contains,
    _counterparty_matches_hint
)

_DUPLICATE_KW: List[str] = [
    "duplicate",
    "twice",
    "double",
    "two times",
    "charged twice",
    "double charge",
    "2 times",
]

def _find_duplicate_pair(history: List[TransactionEntry]) -> Optional[str]:
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

def _score_transaction(
    txn: TransactionEntry,
    amounts: List[float],
    time_window: Optional[Tuple[datetime, datetime]],
    cp_hints: List[str],
    len_history: int,
) -> float:
    score = 0.0
    
    if len_history == 1:
        score += 2.0
        
    if amounts and txn.amount is not None:
        if any(abs(txn.amount - a) < 0.01 for a in amounts):
            score += 5.0
            
    if cp_hints and txn.counterparty:
        if _counterparty_matches_hint(txn.counterparty, cp_hints):
            score += 4.0
            
    if time_window:
        try:
            ts = _parse_timestamp(txn.timestamp)
            start, end = time_window
            if start <= ts <= end:
                score += 2.0
        except (ValueError, TypeError):
            pass
            
    return score

def find_relevant_transaction(
    complaint: str,
    history: List[TransactionEntry],
) -> Optional[str]:
    if not history:
        return None

    now = datetime.now(timezone.utc)
    try:
        latest = max(_parse_timestamp(t.timestamp) for t in history)
        if latest > now - timedelta(days=365):
            now = latest
    except (ValueError, TypeError):
        pass

    if _lower_contains(complaint, _DUPLICATE_KW):
        dup_id = _find_duplicate_pair(history)
        if dup_id is not None:
            return dup_id

    amounts = _extract_amounts(complaint)
    time_window = _extract_time_window(complaint, now)
    cp_hints = _extract_counterparty_hints(complaint)

    scored_candidates = []
    len_history = len(history)
    for txn in history:
        score = _score_transaction(txn, amounts, time_window, cp_hints, len_history)
        if score > 0:
            scored_candidates.append((score, txn))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_txn = scored_candidates[0]

    if best_score < 4.0:
        return None

    if len(scored_candidates) > 1:
        runner_up_score = scored_candidates[1][0]
        if abs(best_score - runner_up_score) < 0.01:
            return None

    return best_txn.transaction_id
