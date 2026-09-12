import pytest
from src.models.classifier import HinglishSentimentClassifier


@pytest.fixture(scope="module")
def classifier():
    return HinglishSentimentClassifier()


# =====================================================================
# 1. Minimum Functionality Tests (MFT): Core linguistic capabilities
# =====================================================================
def test_mft_simple_positive(classifier):
    res = classifier.predict("Yeh movie sach me bohot badhiya hai")
    assert res["sentiment"]["label"] == "POSITIVE"


def test_mft_simple_negative(classifier):
    res = classifier.predict("Yeh product bilkul bekaar aur ghatiya hai")
    assert res["sentiment"]["label"] == "NEGATIVE"


def test_mft_hard_negation(classifier):
    # A positive word 'accha' inverted by 'nahi'
    res = classifier.predict("Yeh bilkul bhi accha nahi hai")
    assert res["sentiment"]["label"] == "NEGATIVE"


# =====================================================================
# 2. Invariance Tests (INV): Spelling perturbations shouldn't flip label
# =====================================================================
@pytest.mark.parametrize("variant", [
    "bohot badhiya phone hai",
    "bht badiya phone hai",
    "bahut badhya phone h",
    "boht badiya phone h",
    "bhoooot badiya phone hai"
])
def test_inv_spelling_perturbation(classifier, variant):
    res = classifier.predict(variant)
    assert res["sentiment"]["label"] == "POSITIVE", f"Failed for variant: {variant}"


@pytest.mark.parametrize("variant", [
    "bilkul bekaar service hai",
    "bilkul bkr service h",
    "bilkul bakwas service hai",
    "bilkul bkwas service h"
])
def test_inv_negative_slang_invariance(classifier, variant):
    res = classifier.predict(variant)
    assert res["sentiment"]["label"] == "NEGATIVE", f"Failed for variant: {variant}"


# =====================================================================
# 3. Directional Expectation Tests (DIR): Modifiers must drop score
# =====================================================================
def test_dir_negative_intensifier(classifier):
    # Base positive sentence
    base = classifier.predict("Phone accha hai")
    # Adding negative negation must drop positive probability
    modified = classifier.predict("Phone accha nahi hai")

    base_pos = base["sentiment"]["probabilities"]["POSITIVE"]
    mod_pos = modified["sentiment"]["probabilities"]["POSITIVE"]

    assert mod_pos < base_pos, f"Positive probability did not decrease: {base_pos} -> {mod_pos}"
    assert modified["sentiment"]["label"] == "NEGATIVE"


def test_dir_sarcasm_inversion(classifier):
    # Literal praise
    literal = classifier.predict("Bohot badiya kaam kiya")
    # Sarcastic praise
    sarcastic = classifier.predict("Wah bhai 3 ghante me cold coffee deliver ki shabaash")

    assert literal["sentiment"]["label"] == "POSITIVE"
    assert sarcastic["sentiment"]["label"] == "NEGATIVE"
    assert sarcastic["sentiment"]["sarcasm_adjusted"] is True
