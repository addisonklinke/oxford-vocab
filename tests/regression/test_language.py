from flashcard_builder.grammar import Word, PartOfSpeech
from flashcard_builder.language import German

from ..controls import configured


class TestGerman:

    def test_verb_irregulars_whitespace_translations(self):
        """Originally these didn't match config because of whitespace

        Only a subset of verbs have this where Google Translate's response
        includes something other than the infinitive. For example

        "zu helfen" -> " helfen" != "helfen: hilft, half, hat geholfen"

        Fixed in deb51dd, 643f164, and 7fa28ae1
        """
        mistakes = [
            Word("aid", PartOfSpeech.VERB),  # helfen
            Word("award", PartOfSpeech.VERB),  # vergeben
            Word("call", PartOfSpeech.VERB),  # anrufen
            Word("retrieve", PartOfSpeech.VERB),  # abrufen
            Word("attack", PartOfSpeech.VERB),  # angreifen
            Word("appoint", PartOfSpeech.VERB),  # ernennen
            Word("invent", PartOfSpeech.VERB),  # erfinden
            Word("seize", PartOfSpeech.VERB),  # ergreifen
            Word("conceal", PartOfSpeech.VERB),  # verbergen
        ]
        cfg = {
            "irregulars": {
                "werben": "wirbt, warb, hat geworben",
                "helfen": "hilft, half, hat geholfen",
                "nehmen": "nimmt, nahm, hat genommen",
                "geben": "gibt, gab, hat gegeben",
                "rufen": "ruft, rief, hat gerufen",
                "greifen": "greift, griff, hat gegriffen",
                "nennen": "nennt, nannte, hat genannt",
                "finden": "findet, fand, hat gefunden",
                "bergen": "birgt, barg, hat geborgen",
            },
        }
        with configured(German(), cfg) as de:
            actual = {m.word: de.get_translation(m).conjugation for m in mistakes}
        expected = {
            "advertise": "wirbt, warb, hat geworben",
            "aid": "hilft, half, hat geholfen",
            "award": "gibt, gab, hat gegeben",
            "call": "ruft, rief, hat gerufen",
            "retrieve": "ruft, rief, hat gerufen",
            "attack": "greift, griff, hat gegriffen",
            "appoint": "nennt, nannte, hat genannt",
            "invent": "findet, fand, hat gefunden",
            "seize": "greift, griff, hat gegriffen",
            "result": "gibt, gab, hat geben",
            "conceal": "birgt, barg, hat geborgen",
        }
        for k, v in actual.items():
            assert v == expected[k], f"Wrong conjugation note for {k}"
