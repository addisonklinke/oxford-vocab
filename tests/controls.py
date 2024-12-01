"""Utilities to control the behavior of various objects for testing purposes"""

from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Dict

from flashcard_builder.language import Language


@contextmanager
def configured(language: Language, cfg: Dict[str, Any]) -> Language:
    """Temporarily override the language config for testing a specific case"""
    original = deepcopy(language.cfg)
    valid_keys = set(original.keys())
    for key in cfg:
        if key not in valid_keys:
            raise ValueError(f"Invalid key: {key}")
    language.cfg.update(cfg)
    yield language
    language.cfg = original
