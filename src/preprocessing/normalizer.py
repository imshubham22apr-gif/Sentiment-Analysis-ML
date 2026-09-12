import re
from typing import Dict, List, Tuple


class HinglishNormalizer:
    """
    Phonetic, slang, and repeated-character normalizer specifically tailored
    for Hinglish (Romanized Hindi-English code-mixed) text.
    """

    # Common transliteration, phonetic, and SMS/chat variations
    SLANG_AND_PHONETIC_MAP: Dict[str, str] = {
        # Positives
        "axha": "accha",
        "acha": "accha",
        "achha": "accha",
        "achaa": "accha",
        "achhaa": "accha",
        "badhya": "badhiya",
        "badiya": "badhiya",
        "badeeya": "badhiya",
        "bdya": "badhiya",
        "zabardast": "zabardast",
        "jabardast": "zabardast",
        "zbrdst": "zabardast",
        "shandar": "shandaar",
        "shandaar": "shandaar",
        "mst": "mast",
        "mastt": "mast",
        "masttt": "mast",
        "osm": "awesome",
        "awsm": "awesome",
        "gud": "good",
        "gd": "good",
        "sahi": "sahi",
        "shi": "sahi",
        "shii": "sahi",

        # Negatives
        "bekar": "bekaar",
        "bkr": "bekaar",
        "bkwas": "bakwas",
        "bakwaas": "bakwas",
        "bkwass": "bakwas",
        "khrb": "kharab",
        "khraab": "kharab",
        "ghatiya": "ghatiya",
        "ghtya": "ghatiya",
        "falthu": "faltu",
        "faltu": "faltu",
        "thanda": "thanda",
        "thnda": "thanda",

        # Intensifiers & Adverbs
        "bht": "bohot",
        "boht": "bohot",
        "bhot": "bohot",
        "bhut": "bohot",
        "bahut": "bohot",
        "bhoot": "bohot",
        "vry": "very",
        "veri": "very",
        "v": "very",
        "jyada": "zyada",
        "jada": "zyada",
        "zyada": "zyada",
        "kam": "kam",

        # Negation markers (Critical for sentiment polarity!)
        "nhi": "nahi",
        "nhin": "nahi",
        "ni": "nahi",
        "nhii": "nahi",
        "na": "na",
        "naa": "na",
        "mat": "mat",
        "mt": "mat",

        # Common pronouns / social / filler words
        "bhaiii": "bhai",
        "bhaai": "bhai",
        "bhay": "bhai",
        "brooo": "bro",
        "yaar": "yaar",
        "yr": "yaar",
        "yrr": "yaar",
        "plz": "please",
        "pls": "please",
        "plzz": "please",
        "thx": "thanks",
        "thnx": "thanks",
        "thanx": "thanks",
        "tq": "thanks",
        "dhanyawad": "thanks",
        "kch": "kuch",
        "kuchh": "kuch",
        "yeh": "yeh",
        "ye": "yeh",
        "woh": "woh",
        "wo": "woh",
        "h": "hai",
        "tha": "tha",
        "thi": "thi",
        "the": "the",
        "kya": "kya",
        "kyu": "kyun",
        "kyun": "kyun",
        "q": "kyun",
        "lekin": "lekin",
        "lkn": "lekin",
        "par": "par",
        "pr": "par",
    }

    def __init__(self, custom_lexicon: Dict[str, str] = None):
        self.lexicon = dict(self.SLANG_AND_PHONETIC_MAP)
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

    def compress_repeated_characters(self, word: str) -> str:
        """
        Compresses elongated characters caused by emotional chat emphasis.
        e.g., 'sooooo' -> 'soo', 'bhaaaai' -> 'bhai', 'mastttt' -> 'mast'
        Allows max 2 consecutive identical characters.
        """
        # Compress 3 or more repeating characters down to 1 if typical, or 2
        # For Hinglish, words like 'accha' have 'cc', but 'bhaaaai' -> 'bhai'
        compressed = re.sub(r'(.)\1{2,}', r'\1', word)
        return compressed

    def normalize_word(self, word: str) -> Tuple[str, bool]:
        """
        Normalizes a single token using compression and dictionary mapping.
        Returns (normalized_word, was_modified).
        """
        clean_word = word.lower().strip()
        compressed = self.compress_repeated_characters(clean_word)

        if compressed in self.lexicon:
            return self.lexicon[compressed], True
        elif clean_word in self.lexicon:
            return self.lexicon[clean_word], True
        elif compressed != clean_word:
            return compressed, True
        return word, False

    def normalize(self, text: str) -> str:
        """
        Full text normalization pipeline:
        1. Preserves punctuation spacing.
        2. Normalizes chat slang, phonetic variants, and elongated tokens.
        3. Standardizes whitespace.
        """
        if not text or not text.strip():
            return ""

        # Normalize emojis/punctuation spacing
        text = re.sub(r'([!?,;:\(\)])', r' \1 ', text)
        tokens = text.split()

        normalized_tokens: List[str] = []
        for token in tokens:
            # Preserve purely punctuation tokens
            if re.match(r'^[!?,;:\(\)\.]+$', token):
                normalized_tokens.append(token)
                continue

            # Separate alphanumeric from attached trailing symbols
            match = re.match(r'^([a-zA-Z0-9_\u0900-\u097F]+)(.*)$', token)
            if match:
                word_part, symbol_part = match.groups()
                norm_word, _ = self.normalize_word(word_part)
                normalized_tokens.append(norm_word + symbol_part)
            else:
                norm_word, _ = self.normalize_word(token)
                normalized_tokens.append(norm_word)

        result = " ".join(normalized_tokens)
        # Clean double spaces and punctuation attachment
        result = re.sub(r'\s+([!?,;:\)])', r'\1', result)
        result = re.sub(r'(\()\s+', r'\1', result)
        result = re.sub(r'\s{2,}', ' ', result).strip()
        return result

    def get_diff_summary(self, raw_text: str) -> Dict[str, any]:
        """Returns comparison metrics between raw and normalized text."""
        normalized = self.normalize(raw_text)
        raw_tokens = raw_text.split()
        norm_tokens = normalized.split()
        
        changes = []
        for r, n in zip(raw_tokens, norm_tokens):
            if r.lower() != n.lower():
                changes.append({"original": r, "normalized": n})

        return {
            "raw_text": raw_text,
            "normalized_text": normalized,
            "changes_count": len(changes),
            "changes": changes,
        }
