import re
from typing import Dict, List, Tuple

# Common English stopwords and functional words
ENGLISH_WORDS = {
    "the", "is", "a", "an", "and", "or", "in", "on", "at", "for", "with",
    "this", "that", "it", "my", "your", "his", "her", "their", "our",
    "was", "were", "are", "be", "been", "have", "has", "had", "do", "does",
    "did", "very", "good", "bad", "phone", "camera", "battery", "service",
    "quality", "screen", "display", "fast", "slow", "happy", "sad", "like",
    "love", "hate", "buy", "product", "delivery", "order", "food", "taste",
    "app", "update", "working", "super", "awesome", "worst", "best", "money",
    "waste", "worth", "time", "hour", "day", "night", "call", "support"
}

# Common Romanized Hindi words (and Devanagari range)
HINDI_WORDS = {
    "hai", "h", "tha", "thi", "the", "yeh", "ye", "woh", "wo", "kya",
    "kaise", "kyun", "kyu", "ka", "ki", "ke", "ko", "se", "me", "mein",
    "par", "pr", "lekin", "lkn", "bhi", "toh", "to", "bohot", "bht", "bahut",
    "acha", "accha", "badhiya", "bekar", "bekaar", "bakwas", "mast",
    "nahi", "nhi", "ni", "mat", "kuch", "sab", "apna", "meri", "mera", "mere",
    "tera", "teri", "tere", "unka", "unki", "unke", "hoga", "hogi", "honge",
    "bhai", "yaar", "yr", "daam", "paisa", "khana", "swaad", "shandaar",
    "zabardast", "bilkul", "waah", "wah", "kripya", "sahi", "kharab", "ghatiya",
    "aur", "ya", "magar", "agar", "jab", "tab", "hum", "tum", "aap", "muje", "mujhe"
}


def tag_token(token: str) -> str:
    """
    Tags token as 'en' (English), 'hi' (Hindi/Romanized Hindi),
    or 'univ' (Punctuation, numbers, universal symbols).
    """
    clean = token.lower().strip()
    if not clean or re.match(r'^[0-9\W_]+$', clean):
        return "univ"

    # Check Devanagari script range
    if any('\u0900' <= char <= '\u097F' for char in clean):
        return "hi"

    if clean in HINDI_WORDS:
        return "hi"
    if clean in ENGLISH_WORDS:
        return "en"

    # Morphological heuristics: e.g. Hindi verb endings (-ta, -ti, -te, -raha, -rahi, -rahe)
    if any(clean.endswith(suffix) for suffix in ["raha", "rahi", "rahe", "unga", "ungi", "enge", "wala", "wali"]):
        return "hi"

    # Default heuristic: if alphabetic and not recognized, treat based on character distribution
    return "en"


def tag_tokens(tokens: List[str]) -> List[Tuple[str, str]]:
    """Tags a list of tokens with language tags."""
    return [(token, tag_token(token)) for token in tokens]


def calculate_cmi(text: str) -> Dict[str, any]:
    """
    Calculates the Code-Mixing Index (CMI) (Gambäck & Das, 2014):
    CMI = 100 * [1 - max(w1, w2) / (N - u)]
    where N is total tokens, u is universal tokens, w1 and w2 are counts of lang1 and lang2.
    Score ranges from 0 (monolingual) to 50 (equal 50-50 code mix).
    """
    tokens = text.split()
    if not tokens:
        return {"cmi": 0.0, "total_tokens": 0, "hi_count": 0, "en_count": 0, "univ_count": 0, "is_code_mixed": False}

    tagged = [(token, tag_token(token)) for token in tokens]
    hi_count = sum(1 for _, tag in tagged if tag == "hi")
    en_count = sum(1 for _, tag in tagged if tag == "en")
    univ_count = sum(1 for _, tag in tagged if tag == "univ")

    eval_tokens = len(tokens) - univ_count
    if eval_tokens <= 0:
        cmi_score = 0.0
    else:
        max_lang = max(hi_count, en_count)
        cmi_score = round(100.0 * (1.0 - (max_lang / eval_tokens)), 2)

    return {
        "cmi": cmi_score,
        "total_tokens": len(tokens),
        "hi_count": hi_count,
        "en_count": en_count,
        "univ_count": univ_count,
        "is_code_mixed": cmi_score > 0.0,
        "tagged_tokens": tagged
    }
