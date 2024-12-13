import pytest

from flashcard_builder.grammar import PartOfSpeech, Word


class TestWord:

    @pytest.fixture(scope="function")
    def noun_word(self):
        return Word("word", PartOfSpeech("n"))

    @pytest.fixture(scope="function")
    def noun_word_with_note(self, noun_word):
        noun_word.note = "note"
        return noun_word

    @pytest.fixture(scope="function")
    def noun_word_with_plural(self, noun_word):
        noun_word.plural_ending = "plural"
        return noun_word

    @pytest.fixture(scope="function")
    def verb_word(self):
        return Word("talk", PartOfSpeech("v"))

    @pytest.fixture(scope="function")
    def verb_word_with_conjugation(self, verb_word):
        verb_word.conjugation = "conjugation"
        return verb_word

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

    def test_format_front(self, noun_word):
        assert noun_word.format_front() == "word [n.]"

    def test_format_front_with_note(self, noun_word_with_note):
        assert noun_word_with_note.format_front() == "word (note) [n.]"

    def test_format_back_noun(self, noun_word):
        assert noun_word.format_back() == "word"

    def test_format_back_noun_with_plural(self, noun_word_with_plural):
        assert noun_word_with_plural.format_back() == "word, plural"

    def test_format_back_verb(self, verb_word):
        assert verb_word.format_back() == "talk"

    def test_format_back_verb_with_conjugation(self, verb_word_with_conjugation):
        assert verb_word_with_conjugation.format_back() == "talk [conjugation]"
