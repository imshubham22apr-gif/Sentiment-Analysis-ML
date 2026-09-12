import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from src import __version__
from src.models.classifier import HinglishSentimentClassifier
from src.preprocessing.normalizer import HinglishNormalizer
from src.api.schemas import (
    SentimentRequest,
    SentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    HealthResponse,
    AspectDetail
)

app = FastAPI(
    title="Hinglish NLP & Sentiment Analysis Microservice",
    description=(
        "Production-grade NLP API for Romanized Hindi-English (Hinglish) code-mixed text. "
        "Includes phonetic normalization, Code-Mixing Index (CMI), Sarcasm detection, "
        "and Aspect-Based Sentiment Analysis (ABSA)."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier = HinglishSentimentClassifier()
normalizer = HinglishNormalizer()


def _format_prediction_response(res: Dict[str, Any]) -> SentimentResponse:
    """Maps internal model output to Pydantic SentimentResponse schema."""
    aspects_dict = {}
    for asp, details in res["aspects"].items():
        aspects_dict[asp] = AspectDetail(
            sentiment=details["sentiment"],
            score=details["score"],
            confidence=details["confidence"],
            clause=details["clause"]
        )

    return SentimentResponse(
        raw_text=res["input"]["raw_text"],
        normalized_text=res["input"]["normalized_text"],
        sentiment_label=res["sentiment"]["label"],
        confidence=res["sentiment"]["confidence"],
        sarcasm_adjusted=res["sentiment"]["sarcasm_adjusted"],
        probabilities=res["sentiment"]["probabilities"],
        linguistics=res["linguistics"],
        sarcasm=res["sarcasm"],
        aspects=aspects_dict,
        latency_ms=res["runtime_metadata"]["latency_ms"],
        engine=res["runtime_metadata"]["engine"]
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns microservice health and active inference engine."""
    return HealthResponse(
        status="healthy",
        engine=classifier.engine_type,
        version=__version__
    )


@app.post("/api/v1/sentiment", response_model=SentimentResponse, tags=["Sentiment"])
def predict_sentiment(request: SentimentRequest):
    """
    Analyzes sentiment of a single Hinglish sentence with linguistic normalization,
    CMI calculation, sarcasm verification, and aspect extraction.
    """
    try:
        raw_res = classifier.predict(request.text)
        return _format_prediction_response(raw_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/v1/sentiment/batch", response_model=BatchSentimentResponse, tags=["Sentiment"])
def predict_sentiment_batch(request: BatchSentimentRequest):
    """
    Batch sentiment analysis for production pipelines with aggregated latency metrics.
    """
    start = time.perf_counter()
    try:
        predictions = []
        for text in request.texts:
            raw_res = classifier.predict(text)
            predictions.append(_format_prediction_response(raw_res))

        total_latency = round((time.perf_counter() - start) * 1000, 2)
        return BatchSentimentResponse(
            predictions=predictions,
            total_processed=len(predictions),
            total_latency_ms=total_latency
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.post("/api/v1/normalize", tags=["Preprocessing"])
def normalize_text(request: SentimentRequest):
    """
    Performs phonetic normalization and slang expansion on raw Hinglish text.
    """
    return normalizer.get_diff_summary(request.text)
