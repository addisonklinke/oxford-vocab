"""End to end workflow for translating words to flashcards."""

import json
import os
from typing import Any, Dict, List

import pandas as pd
import pytest

from flashcard_builder.flashcards import FlashCardBuilder
from flashcard_builder.grammar import PartOfSpeech, Word
from flashcard_builder.language import German

from ..controls import configured


class TestFlashCardBuilderIntegrationGerman:

    @pytest.fixture(scope="class")
    def cfg(self) -> Dict[str, Any]:
        """Override configuration for German language"""
        return {
            "skip": [
                "might [v.]",
            ],
            "irregulars": {
                "geben": "gibt, gab, hat gegeben",
            },
            "plurals": {
                "der Augenblick": "-e",
            },
            "translations": {
                "become [v.]": "werden [wird, wurde, ist geworden]",
                "bad (objectively)[adj.]": "schlecht",
                "bad(subjectively) [adj.]": "schlimm",
            },
        }

    @pytest.fixture(scope="class")
    def words(self) -> List[Word]:
        """Words to translate into flashcards"""
        return [
            # General
            Word("might", PartOfSpeech.VERB),  # Skipped
            Word("bad", PartOfSpeech.ADJECTIVE),  # Regular: to be replaced by disambiguated translation
            Word("become", PartOfSpeech.VERB),  # Has a manual translation (not including disambiguation)

            # Nouns
            # TODO configured plural by association with root word
            Word("moment", PartOfSpeech.NOUN),  # Configured plural (translate -> der Augenblick, die Momente)
            Word("difficulty", PartOfSpeech.NOUN),  # Extract ending: hardcoded rule (keit -> -en)
            Word("address", PartOfSpeech.NOUN),  # Extract ending: exact extension of singular
            Word("advice", PartOfSpeech.NOUN),  # Extract ending: umlaut in plural
            Word("parent", PartOfSpeech.NOUN),  # Extract ending: expecting `None` from irregular plural (involving stem change)

            # Verbs
            Word("abuse", PartOfSpeech.VERB),  # Disambiguated by including English `to` (otherwise gets noun)
            Word("give", PartOfSpeech.VERB),  # Direct match for configured irregular
            Word("spend", PartOfSpeech.VERB),  # Matches configured irregular once prefix is removed
        ]

    def test_to_csv(self, words, cfg, tmp_path):
        """Translate words and save to CSV"""

        # TODO mock dispatch to test returned words with extra spaces
        # TODO mock translate: more than two words in German plural
        # TODO mock translate: German article isn't `die`

        # Run export
        base_file = tmp_path / "flashcards"
        with configured(German(), cfg) as german:
            builder = FlashCardBuilder(words, dest=german)
            builder.to_csv(str(base_file))

        # Load back to disk and compare
        csv_path = base_file.with_suffix(".csv")
        assert os.path.isfile(csv_path), "CSV failed to export"
        sort_key = "front"
        actual = pd.read_csv(csv_path).sort_values(sort_key).reset_index(drop=True)
        rows = [
            {"front": "bad (objectively)", "back": "schlecht", "pos": "adj"},
            {"front": "bad (subjectively)", "back": "schlimm", "pos": "adj"},
            {"front": "become", "back": "werden [wird, wurde, ist geworden]", "pos": "v"},
            {"front": "moment", "back": "der Augenblick, -e", "pos": "n"},
            {"front": "difficulty", "back": "die Schwierigkeit, -en", "pos": "n"},
            {"front": "address", "back": "die Adresse, -n", "pos": "n"},
            {"front": "advice", "back": "der Ratschlag, -̈e", "pos": "n"},
            {"front": "parent", "back": "der Elternteil", "pos": "n"},
            {"front": "abuse", "back": "missbrauchen", "pos": "v"},
            {"front": "give", "back": "geben [gibt, gab, hat gegeben]", "pos": "v"},
            {"front": "ausgeben", "back": "ausgeben [gibt, gab, hat gegeben.]", "pos": "v"},
        ]
        for r in rows:
            r.update({"level": pd.NA})
        expected = pd.DataFrame(rows).sort_values(sort_key).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
