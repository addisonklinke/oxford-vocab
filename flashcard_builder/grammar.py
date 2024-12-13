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


@dataclass
class Word:
    """A word and its associated metadata for a vocabulary list"""

    word: str
    pos: Optional[PartOfSpeech] = None
    plural_ending: Optional[str] = None
    conjugation: Optional[str] = None
    note: Optional[str] = None
    level: Optional[str] = None

    def __hash__(self):
        """Don't differentiate by level or plural ending as these are more error-prone

        Part of speech and note are purposefully used to disambiguate words,
        so these are important to include
        """
        relevant = self.word
        if self.pos:
            relevant += str(self.pos)
        if self.note:
            relevant += self.note
        return hash(relevant)

    def __lt__(self, other):
        """Pandas groupby implicitly requires sortability"""
        return self.word < other.word

    def _format_pos(self):
        return f"[{self.pos}.]"

    def format_front(self) -> str:
        """Formatted for front side of flashcard with POS + note to disambiguate"""
        out = self.word
        if self.note:
            out += f" ({self.note})"
        if self.pos:
            out += f" {self._format_pos()}"
        return out

    def format_back(self) -> str:
        """Formatted for back side of flashcard with plural/conjugation info"""

        # TODO add distinction for plural only nouns (currently in YAML as ~)

        out = self.word
        if self.pos == PartOfSpeech.NOUN and self.plural_ending:
            out += f", {self.plural_ending}"
        if self.pos == PartOfSpeech.VERB and self.conjugation:
            out += f" [{self.conjugation}]"
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
