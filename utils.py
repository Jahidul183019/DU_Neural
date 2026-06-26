import re
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from models import TransactionEntry

# Bengali digit → ASCII digit translation table
_BN_DIGIT_TABLE = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def _has_bengali(text: str) -> bool:
    return bool(re.search(r"[\u0980-\u09FF]", text))

def _has_latin(text: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", text))

def _parse_timestamp(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _normalize_phone(raw: str) -> str:
    if not raw or raw[0].isalpha():
        return raw
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 10:
        if digits.startswith("880") and len(digits) == 13:
            return "0" + digits[3:]
        if digits.startswith("01") and len(digits) == 11:
            return digits
    return raw

_WORD_AMOUNTS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000,
    "\u098f\u0995": 1, "\u09a6\u09c1\u0987": 2, "\u09a4\u09bf\u09a8": 3, "\u099a\u09be\u09b0": 4,
    "\u09aa\u09be\u0981\u099a": 5, "\u099b\u09af\u09bc": 6, "\u099b\u09df": 6, "\u09b8\u09be\u09a4": 7,
    "\u0986\u099f": 8, "\u09a8\u09af\u09bc": 9, "\u09a8\u09df": 9, "\u09a6\u09b6": 10,
    "\u09b6\u09a4": 100, "\u09b9\u09be\u099c\u09be\u09b0": 1000, "\u09b2\u09be\u0996": 100000,
}
_MULTIPLIERS = {"thousand", "hundred", "lakh", "lac",
                "\u09b9\u09be\u099c\u09be\u09b0", "\u09b6\u09a4", "\u09b2\u09be\u0996"}

def _extract_amounts(text: str) -> List[float]:
    normalised = text.translate(_BN_DIGIT_TABLE)
    matches = re.findall(r"(?<!\w)[\d,]+(?:\.\d+)?(?!\w)", normalised)
    amounts: List[float] = []
    for m in matches:
        try:
            val = float(m.replace(",", ""))
            if val > 0:
                amounts.append(val)
        except ValueError:
            continue
    tokens = normalised.lower().split()
    for i, token in enumerate(tokens):
        if token in _WORD_AMOUNTS:
            value = _WORD_AMOUNTS[token]
            if i + 1 < len(tokens) and tokens[i + 1] in _MULTIPLIERS:
                value *= _WORD_AMOUNTS[tokens[i + 1]]
                if value not in amounts:
                    amounts.append(float(value))
            elif token not in _MULTIPLIERS:
                if float(value) not in amounts:
                    amounts.append(float(value))
    return amounts

def _extract_time_window(text: str, now: datetime) -> Optional[Tuple[datetime, datetime]]:
    lower = text.lower()
    if "today" in lower:
        return (now.replace(hour=0, minute=0, second=0, microsecond=0), now)
    if any(kw in lower for kw in ("morning", "afternoon", "evening")):
        return (now - timedelta(hours=24), now)
    if "yesterday" in lower:
        start_of_yesterday = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(hours=24)
        return (start_of_yesterday, start_of_yesterday + timedelta(hours=24))
    if re.search(r"\d{1,2}\s*(?:am|pm)", lower):
        return (now - timedelta(hours=24), now)
    return None

def _extract_counterparty_hints(text: str) -> List[str]:
    hints: List[str] = []
    phones = re.findall(r"(?:\+?880|0)1[3-9]\d{8}", text)
    hints.extend(phones)
    merchants = re.findall(r"(?:MER|MERCHANT)-\w+", text, re.IGNORECASE)
    hints.extend([m.upper() for m in merchants])
    agents = re.findall(r"AGENT-\w+", text, re.IGNORECASE)
    hints.extend([a.upper() for a in agents])
    return hints

def _lower_contains(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(re.search(rf"(?:\b|\s|^){re.escape(kw)}(?:\b|\s|$)", lower) for kw in keywords)

def _counterparty_matches_hint(counterparty: str, hints: List[str]) -> bool:
    cp_norm = _normalize_phone(counterparty)
    for hint in hints:
        hint_norm = _normalize_phone(hint)
        if hint in counterparty or hint_norm in cp_norm or counterparty in hint or cp_norm in hint_norm:
            return True
    return False

def _get_relevant_amount(relevant_txn: Optional[TransactionEntry], complaint: str) -> float:
    if relevant_txn:
        return relevant_txn.amount
    amounts = _extract_amounts(complaint)
    return amounts[0] if len(amounts) == 1 else 0.0

def detect_language(complaint: str) -> str:
    has_bn = _has_bengali(complaint)
    has_lat = _has_latin(complaint)
    if has_bn and has_lat:
        return "mixed"
    if has_bn:
        return "bn"
    return "en"
