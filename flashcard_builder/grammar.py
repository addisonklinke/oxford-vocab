from dataclasses import dataclass
from enum import StrEnum
from typing import Optional

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


@dataclass
class Word:
    """A word and its associated metadata for a vocabulary list"""

    word: str
    pos: PartOfSpeech
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

    def to_key(self):
        """Include POS to allow unique lookup within manual translation configs"""
        return f"{self.word} {self._format_pos()}"
