from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

from .patterns import SERIALIZED_WORD

POS_CHARS = "nvadjcopredt"  # TODO make this a dynamic property of PartOfSpeech


class PartOfSpeech(StrEnum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "adj"
    ADVERB = "adv"
    CONJUNCTION = "conj"
    PREPOSITION = "prep"
    DETERMINER = "det"
    PRONOUN = "pron"


class Word:
    """A word and its associated metadata for a vocabulary list"""

    word: str
    pos: Optional[PartOfSpeech] = None
    plural_ending: Optional[str] = None
    note: Optional[str] = None
    level: Optional[str] = None

    def _format_pos(self):
        return f"[{self.pos}.]"

    def format(self, word_only: bool = False) -> str:
        """Formatted word (note) [pos.]"""

        # TODO add distinction for plural only nouns (currently in YAML as ~)
        # TODO make boolean flags more granular

        out = self.word
        if word_only:
            return out
        if self.note:
            out += f" ({self.note})"
        out += f" {self._format_pos()}"
        return out

    @classmethod
    def from_string(cls, s: str) -> "Word":
        """Parse string from CSV or config YAML"""
        match = SERIALIZED_WORD.match(s)
        if not match:
            raise ValueError(f"Invalid word format: {s}. Must match: {SERIALIZED_WORD.pattern}")
        word, note, pos = match.groups()
        return cls(word, PartOfSpeech(pos), note=note)

    def to_key(self):
        """Include POS to allow unique lookup within manual translation configs"""
        return f"{self.word} {self._format_pos()}"
