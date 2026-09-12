import pytest
from src.models.aspect_extractor import AspectBasedSentimentAnalyzer


@pytest.fixture
def analyzer():
    return AspectBasedSentimentAnalyzer()


def test_absa_single_aspect(analyzer):
    res = analyzer.extract_aspect_sentiment("Phone ka camera bohot zabardast hai")
    assert "camera" in res["aspects"]
    assert res["aspects"]["camera"]["sentiment"] == "POSITIVE"


def test_absa_contrasting_multi_aspect(analyzer):
    # Clause 1: Food +ve, Clause 2: Delivery -ve
    text = "Khana bohot badhiya tha lekin delivery bohot slow thi"
    res = analyzer.extract_aspect_sentiment(text)
    assert "food" in res["aspects"]
    assert "delivery" in res["aspects"]
    assert res["aspects"]["food"]["sentiment"] == "POSITIVE"
    assert res["aspects"]["delivery"]["sentiment"] == "NEGATIVE"


def test_absa_negation_inversion(analyzer):
    text = "Battery backup accha nahi hai"
    res = analyzer.extract_aspect_sentiment(text)
    assert "battery" in res["aspects"]
    assert res["aspects"]["battery"]["sentiment"] == "NEGATIVE"


def test_sarcasm_detection_incongruity(analyzer):
    text = "Wah bhai 3 ghante me cold coffee deliver ki shabaash"
    sarcasm = analyzer.detect_sarcasm(text)
    assert sarcasm["is_sarcastic"] is True
    assert sarcasm["sarcasm_score"] >= 0.8


def test_sarcasm_detection_negative_case(analyzer):
    text = "Khana sach me bohot badhiya aur garam tha"
    sarcasm = analyzer.detect_sarcasm(text)
    assert sarcasm["is_sarcastic"] is False
