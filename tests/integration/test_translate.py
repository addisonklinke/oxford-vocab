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
                "bad (objectively) [adj.]": "schlecht",
                "bad (subjectively) [adj.]": "schlimm",
            },
        }

    @pytest.fixture(scope="class")
    def words(self) -> List[Word]:
        """Words to translate into flashcards"""
        level = "A1"
        return [
            # General
            Word("might", PartOfSpeech.VERB, level=level),  # Skipped
            Word("bad", PartOfSpeech.ADJECTIVE, level=level),  # Regular: to be replaced by disambiguated translation
            Word("become", PartOfSpeech.VERB, level=level),  # Has a manual translation (not including disambiguation)

            # Nouns
            # TODO configured plural by association with root word
            Word("moment", PartOfSpeech.NOUN, level=level),  # Configured plural (translate -> der Augenblick, die Momente)
            Word("difficulty", PartOfSpeech.NOUN, level=level),  # Extract ending: hardcoded rule (keit -> -en)
            Word("address", PartOfSpeech.NOUN, level=level),  # Extract ending: exact extension of singular
            Word("art", PartOfSpeech.NOUN, level=level),  # Extract ending: umlaut in plural
            Word("parent", PartOfSpeech.NOUN, level=level),  # Extract ending: expecting `None` from irregular plural (involving stem change)

            # Verbs
            Word("abuse", PartOfSpeech.VERB, level=level),  # Disambiguated by including English `to` (otherwise gets noun)
            Word("give", PartOfSpeech.VERB, level=level),  # Direct match for configured irregular
            Word("spend", PartOfSpeech.VERB, level=level),  # Matches configured irregular once prefix is removed

            # TODO other POS to test lowercasing and article removal
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
        actual = pd.read_csv(csv_path)
        sort_key = "en"
        assert sort_key in actual.columns, f"Missing sort key: {sort_key}"
        actual = actual.sort_values(sort_key).reset_index(drop=True)
        expected = pd.DataFrame(
            [
                ("bad (objectively) [adj.]", "schlecht", "adj"),
                ("bad (subjectively) [adj.]", "schlimm", "adj"),
                ("become [v.]", "werden [wird, wurde, ist geworden]", "v"),
                ("moment [n.]", "der Augenblick, -e", "n"),
                ("difficulty [n.]", "die Schwierigkeit, -en", "n"),
                ("address [n.]", "die Adresse, -n", "n"),
                ("art [n.]", "die Kunst, -̈e", "n"),
                ("parent [n.]", "der Elternteil", "n"),
                ("abuse [v.]", "missbrauchen", "v"),
                ("give [v.]", "geben [gibt, gab, hat gegeben]", "v"),
                ("spend [v.]", "ausgeben [gibt, gab, hat gegeben]", "v"),
            ],
            columns=["en", German.name, "pos"],
        )
        expected["level"] = "A1"
        expected = expected.sort_values(sort_key).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
