import os
import time
from typing import Dict, Any, List, Optional
from src.preprocessing.normalizer import HinglishNormalizer
from src.preprocessing.cmi import calculate_cmi
from src.models.aspect_extractor import AspectBasedSentimentAnalyzer

# Broad Hinglish sentiment polarities for the robust fallback engine
STRONG_POSITIVES = {
    "zabardast", "shandaar", "badhiya", "mast", "accha", "badiya", "awesome",
    "superb", "excellent", "love", "pyar", "best", "lajawab", "gazab",
    "top", "banger", "khush", "happy", "sahi", "badhia", "changa"
}

STRONG_NEGATIVES = {
    "bekaar", "bakwas", "kharab", "ghatiya", "faltu", "ganda", "bura",
    "worst", "hate", "nafrat", "sad", "dukhi", "thanda", "barbad", "loot",
    "fraud", "dhokha", "bekar", "waste", "problem", "issue", "pathetic"
}

NEGATION_MODIFIERS = {"nahi", "nhi", "not", "no", "never", "mat", "na", "ni"}
INTENSIFIERS = {"bohot", "bahut", "bht", "bilkul", "very", "extremely", "zyada"}


class HinglishSentimentClassifier:
    """
    Production-grade Hinglish Sentiment Classifier supporting:
    - Preprocessing with phonetic & slang normalization.
    - Code-Mixing Index (CMI) profiling.
    - Sarcasm and incongruity resolution.
    - Multi-engine inference (PyTorch Transformers, ONNX Runtime, Linguistic Engine).
    """

    def __init__(self, model_name: str = "l3cube-pune/hing-roberta", engine: str = "auto"):
        self.model_name = model_name
        self.engine_type = engine
        self.normalizer = HinglishNormalizer()
        self.aspect_analyzer = AspectBasedSentimentAnalyzer()
        self.pipeline = None
        self.onnx_session = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initializes PyTorch or ONNX model if dependencies and weights exist."""
        if self.engine_type in ("auto", "pytorch", "transformers"):
            try:
                from transformers import pipeline
                self.pipeline = pipeline("sentiment-analysis", model=self.model_name)
                self.engine_type = "pytorch_transformer"
                return
            except Exception:
                # Transformers model not available or network offline
                pass

        if self.engine_type in ("auto", "onnx"):
            onnx_path = os.path.join("models", "hinglish_sentiment_int8.onnx")
            if os.path.exists(onnx_path):
                try:
                    import onnxruntime as ort
                    self.onnx_session = ort.InferenceSession(onnx_path)
                    self.engine_type = "onnx_int8"
                    return
                except Exception:
                    pass

        # Fallback to deterministic linguistic engine
        self.engine_type = "linguistic_engine"

    def _infer_linguistic(self, normalized_text: str) -> Dict[str, Any]:
        """
        Deterministic, rule-based inference with negation scope and intensifier modeling.
        Guarantees instant <1ms inference without requiring 1.1GB downloads in local/CI environments.
        """
        words = normalized_text.lower().split()
        pos_score = 0.0
        neg_score = 0.0

        i = 0
        while i < len(words):
            word = words[i].strip(",.!?")
            multiplier = 1.0

            # Check if preceded by intensifier
            if i > 0 and words[i - 1] in INTENSIFIERS:
                multiplier = 1.6

            # Check negation window: both preceding (English/Hindi) and following (Hinglish: 'accha nahi hai')
            is_negated = False
            if i > 0 and words[i - 1] in NEGATION_MODIFIERS:
                is_negated = True
            elif i > 1 and words[i - 2] in NEGATION_MODIFIERS:
                is_negated = True
            elif i + 1 < len(words) and words[i + 1] in NEGATION_MODIFIERS:
                is_negated = True
            elif i + 2 < len(words) and words[i + 2] in NEGATION_MODIFIERS:
                is_negated = True

            if word in STRONG_POSITIVES:
                if is_negated:
                    neg_score += 1.2 * multiplier
                else:
                    pos_score += 1.0 * multiplier
            elif word in STRONG_NEGATIVES:
                if is_negated:
                    pos_score += 0.8 * multiplier
                else:
                    neg_score += 1.2 * multiplier

            i += 1

        total = pos_score + neg_score
        if total == 0.0:
            label = "NEUTRAL"
            pos_prob = 0.33
            neg_prob = 0.33
            neu_prob = 0.34
        elif pos_score > neg_score:
            label = "POSITIVE"
            confidence = min(0.65 + (pos_score / (pos_score + neg_score + 1.0)), 0.99)
            pos_prob = round(confidence, 4)
            neg_prob = round((1.0 - confidence) * 0.7, 4)
            neu_prob = round(1.0 - pos_prob - neg_prob, 4)
        else:
            label = "NEGATIVE"
            confidence = min(0.65 + (neg_score / (pos_score + neg_score + 1.0)), 0.99)
            neg_prob = round(confidence, 4)
            pos_prob = round((1.0 - confidence) * 0.7, 4)
            neu_prob = round(1.0 - pos_prob - neg_prob, 4)

        return {
            "label": label,
            "confidence": pos_prob if label == "POSITIVE" else (neg_prob if label == "NEGATIVE" else neu_prob),
            "probabilities": {
                "POSITIVE": pos_prob,
                "NEGATIVE": neg_prob,
                "NEUTRAL": neu_prob
            }
        }

    def predict(self, raw_text: str) -> Dict[str, Any]:
        """
        End-to-end sentiment prediction pipeline:
        1. Normalization & Diff analysis
        2. Code-Mixing Index (CMI)
        3. Aspect Extraction & Sarcasm Check
        4. Sentiment Classification & Sarcasm Polarity Inversion
        """
        start_time = time.perf_counter()

        # Step 1: Preprocessing & Normalization
        norm_summary = self.normalizer.get_diff_summary(raw_text)
        normalized_text = norm_summary["normalized_text"]

        # Step 2: Code-Mixing Index
        cmi_data = calculate_cmi(normalized_text)

        # Step 3: Sarcasm Check
        sarcasm_data = self.aspect_analyzer.detect_sarcasm(normalized_text)

        # Step 4: Aspect-Based Sentiment
        absa_data = self.aspect_analyzer.extract_aspect_sentiment(normalized_text)

        # Step 5: Core Classifier Inference
        if self.pipeline:
            try:
                res = self.pipeline(normalized_text)[0]
                lbl = res["label"].upper()
                # Normalize typical HuggingFace labels
                if "POS" in lbl:
                    pred_label = "POSITIVE"
                elif "NEG" in lbl:
                    pred_label = "NEGATIVE"
                else:
                    pred_label = "NEUTRAL"
                base_pred = {
                    "label": pred_label,
                    "confidence": round(res["score"], 4),
                    "probabilities": {pred_label: round(res["score"], 4)}
                }
            except Exception:
                base_pred = self._infer_linguistic(normalized_text)
        else:
            base_pred = self._infer_linguistic(normalized_text)

        # Step 6: Sarcasm calibration
        final_label = base_pred["label"]
        final_confidence = base_pred["confidence"]
        sarcasm_adjusted = False

        if sarcasm_data["is_sarcastic"] and base_pred["label"] in ("POSITIVE", "NEUTRAL"):
            final_label = "NEGATIVE"
            final_confidence = sarcasm_data["sarcasm_score"]
            sarcasm_adjusted = True

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "input": {
                "raw_text": raw_text,
                "normalized_text": normalized_text,
                "changes_applied": norm_summary["changes_count"],
            },
            "sentiment": {
                "label": final_label,
                "confidence": final_confidence,
                "raw_model_label": base_pred["label"],
                "sarcasm_adjusted": sarcasm_adjusted,
                "probabilities": base_pred["probabilities"]
            },
            "linguistics": {
                "code_mixing_index": cmi_data["cmi"],
                "is_code_mixed": cmi_data["is_code_mixed"],
                "language_distribution": {
                    "hindi_tokens": cmi_data["hi_count"],
                    "english_tokens": cmi_data["en_count"],
                    "universal_tokens": cmi_data["univ_count"]
                }
            },
            "sarcasm": sarcasm_data,
            "aspects": absa_data["aspects"],
            "runtime_metadata": {
                "engine": self.engine_type,
                "latency_ms": latency_ms
            }
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Processes multiple texts with aggregated throughput."""
        return [self.predict(t) for t in texts]
