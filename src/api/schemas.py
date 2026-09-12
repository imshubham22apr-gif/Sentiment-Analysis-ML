from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Input text in Hinglish, Hindi, or English",
        examples=["Khana badhiya tha lekin delivery bohot slow thi!"]
    )


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch of texts to analyze concurrently"
    )


class LinguisticsProfile(BaseModel):
    code_mixing_index: float
    is_code_mixed: bool
    language_distribution: Dict[str, int]


class SarcasmProfile(BaseModel):
    is_sarcastic: bool
    sarcasm_score: float
    explanation: str


class AspectDetail(BaseModel):
    sentiment: str
    score: float
    confidence: float
    clause: str


class SentimentResponse(BaseModel):
    raw_text: str
    normalized_text: str
    sentiment_label: str
    confidence: float
    sarcasm_adjusted: bool
    probabilities: Dict[str, float]
    linguistics: LinguisticsProfile
    sarcasm: SarcasmProfile
    aspects: Dict[str, AspectDetail]
    latency_ms: float
    engine: str


class BatchSentimentResponse(BaseModel):
    predictions: List[SentimentResponse]
    total_processed: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    engine: str
    version: str
