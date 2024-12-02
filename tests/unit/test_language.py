import pytest

from flashcard_builder.language import Language, German

from ..controls import configured


class TestGerman:

    @pytest.fixture(scope="function")
    def german(self):
        """Remove the auto-imported config to avoid side-effects on tests"""
        de = German()
        de.cfg = {}
        de.init_missing_cfg_keys()
        return de

    @pytest.mark.parametrize("infinitive, note, prefix", [
        ("sein", "ist, war, ist gewesen", ""),  # Exact match
        ("rufen", "rief, hat angerufen", "ab"),  # Separable prefix
        ("finden", "fand, hat gefunden", "er"),  # Inseparable prefix
    ])
    def test_extract_irregular_verb_forms(self, german, infinitive, note, prefix):
        with configured(german, {"irregulars": {infinitive: note}}) as german:
            assert german.extract_irregular_verb_forms(
                infinitive_en="",  # Shouldn't matter for configured cases
                infinitive_native=prefix + infinitive,
            ) == note

    @pytest.mark.parametrize("singular", ["Belohnung", "Freiheit", "Tatigkeit", "Transaktion", "Kollision"])
    def test_extract_plural_ending_feminine_hardcoded(self, german, singular):
        """Configured suffixes should always result in -en"""
        assert german.extract_plural_ending(
            singular=f"die {singular}",
            plural=""  # Shouldn't matter for hardcoded cases
        ) == "-en"

    def test_get_noun_translation_configured_plural(self, german, mocker):
        """Plural endings are extracted from the configured list instead of using extract method"""

        # Without any config settings this should fail to find a plural
        # Because the correct `Eltern` is not a suffix of `Elternteil`
        spy = mocker.spy(german, "extract_plural_ending")
        english = "parent"
        singular = "der Elternteil"
        plural_ending = "Eltern"
        assert german._get_noun_translation(english).word == singular
        spy.assert_called_once_with(singular=singular, plural=f"die {plural_ending}")

        # With the correct config setting, the plural should be found
        with configured(german, {"plurals": {singular: plural_ending}}) as german:
            word = german._get_noun_translation(english)
        assert word.word == singular
        assert word.plural_ending == plural_ending

    def test_remove_umlauts(self, german):
        """Umlaut characters are replaced with their base form"""
        assert german.remove_umlauts("äöü") == "aou"
