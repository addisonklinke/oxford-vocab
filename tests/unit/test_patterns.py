import pytest

import flashcard_builder.patterns as p


class TestSerializedWord:
    """Parses the on-disk representation of a Word object as: word (note) [pos.]"""

    @pytest.mark.xfail(reason="Not implemented", run=False)
    def test_word_only(self):
        match = p.SERIALIZED_WORD.match("word")
        assert match.groups() == ("word", None, None)

    def test_word_and_pos(self):
        match = p.SERIALIZED_WORD.match("word [n.]")
        assert match.groups() == ("word", None, "n")

    def test_word_pos_and_note(self):
        match = p.SERIALIZED_WORD.match("word (note) [n.]")
        assert match.groups() == ("word", "note", "n")

    def test_word_pos_and_multiword_note(self):
        note = "note with spaces, commas, and other punctuation"
        match = p.SERIALIZED_WORD.match(f"word ({note}) [n.]")
        assert match.groups() == ("word", note, "n")
