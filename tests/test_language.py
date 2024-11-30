from contextlib import contextmanager
from copy import deepcopy
from typing import Dict

import pytest

from flashcard_builder.language import Language, German


@contextmanager
def with_config(language: Language, config: Dict[str, str]) -> Language:
    """Temporarily override the language config for testing a specific case"""
    original = deepcopy(language.config)
    valid_keys = set(original.keys())
    for key in config:
        if key not in valid_keys:
            raise ValueError(f"Invalid key: {key}")
    language.config.update(config)
    yield language
    language.config = original


class TestGerman:

    @pytest.fixture(scope="function")
    def german(self):
        return German()

    def test_remove_umlauts(self, german):
        """Umlaut characters are replaced with their base form"""
        assert german.remove_umlauts("äöü") == "aou"
