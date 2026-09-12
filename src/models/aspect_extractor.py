import re
from typing import Dict, List, Any, Optional

DEFAULT_ASPECTS = {
    "camera": ["camera", "photo", "pic", "picture", "selfie", "clarity", "lens", "video", "sensor", "cam"],
    "battery": ["battery", "charging", "charger", "backup", "drain", "mah", "discharge", "power"],
    "delivery": ["delivery", "rider", "courier", "package", "parcel", "time", "late", "delay", "pahuncha", "shipping"],
    "food": ["food", "khana", "taste", "swaad", "portion", "quantity", "dish", "meal", "biryani", "roti", "chai", "pizza"],
    "service": ["service", "staff", "behavior", "support", "helpdesk", "customer care", "waiter", "attitude", "service"],
    "price": ["price", "paisa", "daam", "rate", "cost", "cheap", "expensive", "sasta", "mehnga", "worth", "value", "rupaye"],
    "performance": ["performance", "speed", "lag", "hang", "processor", "ram", "gaming", "smooth", "slow", "fast"],
    "display": ["screen", "display", "amoled", "brightness", "panel", "touch", "refresh rate", "hz"]
}

POSITIVE_LEXICON = {
    "accha", "acha", "badhiya", "zabardast", "shandaar", "mast", "awesome",
    "good", "great", "excellent", "superb", "fast", "smooth", "crisp",
    "best", "love", "pasand", "sahi", "perfect", "worth", "clear", "top"
}

NEGATIVE_LEXICON = {
    "bekaar", "bekar", "bakwas", "kharab", "ghatiya", "faltu", "slow",
    "drain", "late", "hang", "bad", "worst", "horrible", "thanda", "tuta",
    "bura", "hate", "waste", "barbad", "problem", "issue", "poor", "loss"
}

NEGATION_TERMS = {"nahi", "nhi", "not", "no", "never", "na", "mat", "ni"}

SARCASM_POSITIVE_CUES = [
    "wah", "waah", "shabaash", "shabash", "kya baat", "great job",
    "bohot khoob", "superb", "congratulations", "mubarak", "kamaal"
]

SARCASM_NEGATIVE_TRIGGERS = [
    "ghante", "ghanta", "hours", "late", "cold", "thanda", "tuta", "broken",
    "kharab", "never", "bilkul bekaar", "paisa barbad", "loot", "dhokha", "bekaar"
]


class AspectBasedSentimentAnalyzer:
    """
    Extracts fine-grained aspect-sentiment pairs and detects code-mixed sarcasm
    specifically structured for Hinglish texts.
    """

    def __init__(self, aspects: Optional[Dict[str, List[str]]] = None):
        self.aspects = aspects or DEFAULT_ASPECTS

    def _split_into_clauses(self, text: str) -> List[str]:
        """Splits sentences into sub-clauses based on conjunctives and punctuation."""
        # Split on conjunctives like 'lekin', 'par', 'but', 'aur', 'and', ',', ';', '.', '!'
        delimiters = r'[,;!?]|\s+(?:lekin|par|pr|but|however|aur|and|though|yet)\s+'
        clauses = re.split(delimiters, text, flags=re.IGNORECASE)
        return [c.strip() for c in clauses if c.strip()]

    def extract_aspect_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Extracts aspects mentioned and assigns sentiment polarity to each aspect.
        """
        clauses = self._split_into_clauses(text)
        detected_aspects: Dict[str, Dict[str, Any]] = {}

        for clause in clauses:
            words = [w.lower().strip(",.!?") for w in clause.split()]
            
            # Find aspects in this clause
            clause_aspects = []
            for aspect_name, keywords in self.aspects.items():
                for kw in keywords:
                    if kw in words or any(kw in w for w in words):
                        clause_aspects.append(aspect_name)
                        break

            if not clause_aspects:
                continue

            # Analyze sentiment of the clause
            has_negation = any(w in NEGATION_TERMS for w in words)
            pos_score = sum(1 for w in words if w in POSITIVE_LEXICON)
            neg_score = sum(1 for w in words if w in NEGATIVE_LEXICON)

            # Polarity calculation with negation flip
            raw_polarity = pos_score - neg_score
            if has_negation:
                raw_polarity = -raw_polarity if raw_polarity != 0 else -1

            if raw_polarity > 0:
                sentiment = "POSITIVE"
                confidence = round(min(0.70 + (0.15 * pos_score), 0.98), 2)
            elif raw_polarity < 0:
                sentiment = "NEGATIVE"
                confidence = round(min(0.70 + (0.15 * neg_score), 0.98), 2)
            else:
                sentiment = "NEUTRAL"
                confidence = 0.50

            for aspect in clause_aspects:
                detected_aspects[aspect] = {
                    "sentiment": sentiment,
                    "score": round(raw_polarity / max(abs(raw_polarity) if raw_polarity != 0 else 1, 1), 2),
                    "confidence": confidence,
                    "clause": clause
                }

        return {
            "aspects_found": len(detected_aspects),
            "aspects": detected_aspects,
            "overall_clauses_analyzed": len(clauses)
        }

    def detect_sarcasm(self, text: str) -> Dict[str, Any]:
        """
        Detects Hinglish sarcasm / irony via sentiment incongruity.
        e.g., 'Wah bhai, 3 ghante me cold coffee deliver ki, shabaash.'
        """
        text_lower = text.lower()
        has_pos_cue = any(re.search(rf"\b{re.escape(cue)}\b", text_lower) for cue in SARCASM_POSITIVE_CUES)
        has_neg_trigger = any(re.search(rf"\b{re.escape(trigger)}\b", text_lower) for trigger in SARCASM_NEGATIVE_TRIGGERS)

        # Check for delay numbers with praise (e.g., '3 ghante', '2 din')
        has_delay_pattern = bool(re.search(r'\b\d+\s*(?:ghante|hours|days|din)\b', text_lower))

        is_sarcastic = False
        confidence = 0.0
        reason = "Normal statement"

        if has_pos_cue and (has_neg_trigger or has_delay_pattern):
            is_sarcastic = True
            confidence = 0.88
            reason = "Surface praise contradicted by negative outcome / excessive delay (Sentiment Incongruity)"
        elif "bilkul bekaar" in text_lower and ("shabaash" in text_lower or "wah" in text_lower):
            is_sarcastic = True
            confidence = 0.95
            reason = "Explicit superlative mockery detected"

        return {
            "is_sarcastic": is_sarcastic,
            "sarcasm_score": confidence,
            "explanation": reason
        }
