import pytest
from src.preprocessing.normalizer import HinglishNormalizer
from src.preprocessing.cmi import calculate_cmi, tag_token


@pytest.fixture
def normalizer():
    return HinglishNormalizer()


def test_repeated_character_compression(normalizer):
    assert normalizer.compress_repeated_characters("bhaaaai") == "bhai"
    assert normalizer.compress_repeated_characters("mastttt") == "mast"
    assert normalizer.compress_repeated_characters("soooo") == "so"


def test_phonetic_and_slang_mapping(normalizer):
    assert normalizer.normalize("ye product bht axha h") == "yeh product bohot accha hai"
    assert normalizer.normalize("bkr service h") == "bekaar service hai"
    assert normalizer.normalize("plz call me") == "please call me"


def test_negation_particle_preservation(normalizer):
    norm = normalizer.normalize("ye khana acha nhi h")
    assert "nahi" in norm
    assert "accha" in norm


def test_cmi_calculation_monolingual():
    res_en = calculate_cmi("this is a great phone with awesome battery")
    assert res_en["cmi"] == 0.0
    assert not res_en["is_code_mixed"]

    res_hi = calculate_cmi("yeh mera khana hai aur bohot badhiya hai")
    assert res_hi["cmi"] == 0.0


def test_cmi_calculation_code_mixed():
    res = calculate_cmi("bhai phone ka camera mast hai")
    assert res["is_code_mixed"]
    assert res["cmi"] > 0.0
    assert res["hi_count"] > 0
    assert res["en_count"] > 0
