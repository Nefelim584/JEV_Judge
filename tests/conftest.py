import pytest

BASE_MODEL = "answerdotai/ModernBERT-large"


@pytest.fixture(scope="session")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL)
    except OSError as e:  # not cached and no network
        pytest.skip(f"tokenizer {BASE_MODEL} unavailable: {e}")
