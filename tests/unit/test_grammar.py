import pytest

from flashcard_builder.grammar import Word


class TestWord:

    def test_from_string_pos(self):
        word = Word.from_string("word [n.]")
        assert word.word == "word"
        assert word.pos == "n"
        assert word.note is None

    def test_from_string_pos_and_note(self):
        word = Word.from_string("word (note) [n.]")
        assert word.word == "word"
        assert word.pos == "n"
        assert word.note == "note"

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid word format"):
            Word.from_string("1 word [n.]")
