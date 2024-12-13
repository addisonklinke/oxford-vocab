from collections import defaultdict
from functools import partial
import os
import re
from typing import Callable, Dict, List, Optional, Tuple
import warnings

import inflect
from googletrans import Translator
import mlconjug3
from mlconjug3 import Conjugator
import pandas as pd
from sklearn.base import InconsistentVersionWarning
import spacy
import yaml

from .grammar import PartOfSpeech, Word
from .patterns import WHITESPACE_STRIP

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # mlconjug3
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using")  # spacy/pytorch
conjugator = Conjugator(language="en")
translator = Translator()
try:
    nlp = spacy.load("en_core_web_trf")  # Recommended for best accuracy by https://spacy.io/models
except OSError as exc:
    raise RuntimeError("Missing spaCy model: run `python -m spacy download <model_name>` to fix") from exc


class Language:
    """Base class for handling translations

    Manually defined translations from a config YAML are applied in a few locations
        * `.get_translation()`
            * Checks `cfg["translations"]` before doing anything else
            * Skips anything in `cfg["skip"]` (based on word and POS)
        * `_get_noun_translation()` gives preference to `cfg["plurals"]`
        * `_get_verb_translation()` gives preference to `cfg["irregulars"]`
    """

    name: str
    noun_article: Optional[str] = "the"
    SKIP_KEY = "skip"
    TRANSLATIONS_KEY = "translations"
    IRREGULARS_KEY = "irregulars"
    PLURALS_KEY = "plurals"

    def __init__(self):
        """Load manually defined translations"""

        # Convert config YAML to attribute
        # TODO use directional name for config files to indicate source language
        # TODO consider making config its own class
        cfg_path = os.path.join(os.path.dirname(__file__), f"cfg/{self.name}.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}
        self.init_missing_cfg_keys()

        # Parse manually defined translations for disambiguating notes and POS
        self.ambiguous_words = self.get_ambiguous_words()

    def _get_noun_translation(self, english: str) -> Word:
        # TODO handle bifurcated masculine/feminine nouns (i.e. [stem]erin, -nen)
        if self.noun_article:
            english = f"{self.noun_article} {english}"
        translation = en.translate_to(english, dest=self.name)
        if translation in self.cfg[self.PLURALS_KEY]:
            ending = self.cfg[self.PLURALS_KEY][translation]
        else:
            plural = en.translate_to(en.pluralize(english), dest=self.name)
            ending = self.extract_plural_ending(translation, plural)
        return Word(translation, PartOfSpeech.NOUN, plural_ending=ending)

    def _get_other_translation(self, english: str, pos: PartOfSpeech) -> Word:
        """Wrapper around `translate_from()` for compatibility with `Word` return type"""
        translation = self.translate_from(english, src="en")
        return Word(translation, pos)

    def _get_verb_translation(self, english_infinitive: str) -> Word:
        # FIXME returning a lot of capitalized words
        translated_infinitive_prefix = en.translate_to("to", dest=self.name).lower()
        translation = en.translate_to(
            text="to " + english_infinitive,  # Use infinitive to ensure ambiguous words aren't treated as nouns
            dest=self.name
        )
        translation = translation.lower().replace(translated_infinitive_prefix, "")
        if translation in self.cfg[self.IRREGULARS_KEY]:
            note = self.cfg[self.IRREGULARS_KEY][translation]
        else:
            note = self.extract_irregular_verb_forms(english_infinitive, translation)
        return Word(translation, PartOfSpeech.VERB, conjugation=note)

    def init_missing_cfg_keys(self) -> None:
        """Fill missing keys with empty dictionaries so methods can assume they exist"""
        for key in (self.SKIP_KEY, self.TRANSLATIONS_KEY, self.IRREGULARS_KEY, self.PLURALS_KEY):
            if (
                key not in self.cfg
                or self.cfg[key] is None
            ):
                self.cfg[key] = {}

    def conjugate(self, infinitive: str, tense: str, mood: str, person: str) -> str:
        """Return verb's conjugation in this language"""
        if self.name not in mlconjug3.LANGUAGES:
            # TODO revisit after https://github.com/Ars-Linguistica/mlconjug3/issues/331
            raise NotImplementedError(f"Language {self.name} not supported by mlconjug3")
        verb = conjugator.conjugate(infinitive)
        return verb[mood][tense][person]  # May raise KeyError for invalid parameters

    @staticmethod
    def conditionally_case(text: str) -> str:
        raise NotImplementedError

    def disambiguate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Disambiguate translations based on manual notes"""
        assert "en" in df.columns, "English column must be present"
        assert self.name in df.columns, f"{self.name} column must be present"

        # Get rid of existing translations
        to_remove = []
        english_words = {word.word for word in df.en}
        for i, row in df.iterrows():
            if (row.en.word, row.en.pos) in self.ambiguous_words:
                to_remove.append(i)
        df.drop(to_remove, axis=0, inplace=True)

        # Replace with manual translations
        new_rows = []
        for (english, pos), translations in self.ambiguous_words.items():
            if english not in english_words:
                continue  # Doesn't apply to this vocab list
            for note, translation in translations:
                new_rows.append({
                    "en": Word(english, PartOfSpeech(pos), note=note),
                    self.name: Word(translation),
                })
        assert len(new_rows) >= len(to_remove), "Failed to replace all ambiguous translations"
        return pd.concat([df, pd.DataFrame(new_rows)])

    def extract_irregular_verb_forms(self, infinitive_en: str, infinitive_native: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def extract_plural_ending(self, singular: str, plural: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def get_ambiguous_words(self) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """Get manually defined translations for ambiguous words"""
        ambiguous_words = defaultdict(list)
        for english_str, translation in self.cfg[self.TRANSLATIONS_KEY].items():
            word = Word.from_string(english_str)
            if word.note:
                ambiguous_words[(word.word, word.pos)].append((word.note, translation))
        return ambiguous_words

    def get_translation(self, word: Word) -> Optional[Word]:
        """Translate a word from English into this language"""

        # TODO parameterize the source language so this can be used for any language pair

        k = word.to_key()
        if k in self.cfg[self.SKIP_KEY]:
            return None
        manual_translation = self.cfg[self.TRANSLATIONS_KEY].get(k)
        if manual_translation:
            return Word(manual_translation, pos=word.pos)
        method_map = {
            PartOfSpeech.NOUN: self._get_noun_translation,
            PartOfSpeech.VERB: self._get_verb_translation,
        }
        method: Callable[[str], Word] = method_map.get(word.pos, partial(self._get_other_translation, pos=word.pos))
        translation = method(word.word)
        translation.word = WHITESPACE_STRIP.sub("", translation.word)
        return translation

    def pluralize(self, noun: str) -> str:
        if self.name != "en":
            raise NotImplementedError(f"Not setup to pluralize lang={self.name}")
        engine = inflect.engine()
        return engine.plural_noun(noun)

    def translate_from(self, text: str, src: str) -> str:
        """Translate text from another language to this one"""
        return translator.translate(text, src=src, dest=self.name).text

    def translate_to(self, text: str, dest: str) -> str:
        """Translate text from this language into another"""
        return translator.translate(text, src=self.name, dest=dest).text


class English(Language):
    """Support for several extra operations coming from English-only NLP libraries"""

    name = "en"

    def conjugate(self, infinitive: str, tense: str, mood: str, person: str) -> str:
        """Set default to present tense 3rd person singular"""
        return super().conjugate(infinitive, tense, mood, person)

    def get_translation(self, english: Word) -> Word:
        return english

    @staticmethod
    def conditionally_case(text: str) -> str:
        """Adjust casing so only proper nouns are capitalized

        From discussion here: https://stackoverflow.com/a/63382009/7446465
        Sounds like NLTK only works well when capitalization already follows convention
        """
        # TODO spacy actually supports a lot of non-English languages so this could be upstreamed into the base class
        #  Just need to map `self.name` to the right NLP model to load
        # TODO handle start of sentence capitalization and first person pronouns
        doc = nlp(text.lower())
        fixed = []
        for i, tok in enumerate(doc):
            if (
                i + 1 == len(doc)
                or i < len(doc) - 1 and doc[i + 1].pos_ == "PUNCT"
            ):
                delimeter = ""
            else:
                delimeter = " "
            token = tok.text.capitalize() if tok.pos_ == "PROPN" else tok.text
            fixed.append(token + delimeter)
        return "".join(fixed)


class French(Language):
    name = "fr"
    noun_article = "a"  # Better for getting genders


class German(Language):

    name = "de"

    # https://coffeebreaklanguages.com/2024/06/making-sense-of-german-separable-verbs-a-guide-for-learners/
    separable_prefixes = (
        "ab",  # FIXME abrufen
        "an",  # FIXME not working for anstreben, angreifen
        "auf",  # FIXME aufrufen
        "aus",
        "bei",
        "ein",
        "mit",
        "nach",  # But this works for nachkommen
        "um",
        "vor",
        "zu",
    )
    inseparable_prefixes = (
        "be",
        "emp",
        "ent",  # FIXME not working for entlassen
        "er",   # FIXME not working for ernennen, erfinden, ergreifen, ergeben
        "ver",  # FIXME not working for vergeben, verbergen
        "zer",
    )

    def _get_noun_translation(self, english: str) -> Word:
        """Make sure article is lowercase"""
        translation = super()._get_noun_translation(english)
        article, *noun = translation.word.split()
        article = article.lower()
        noun[0] = noun[0].capitalize()
        translation.word = article + " " + " ".join(noun)
        return translation

    def _get_verb_translation(self, english_infinitive: str) -> Word:
        translation = super()._get_verb_translation(english_infinitive)
        translation.word = re.sub("^u?m ", "", translation.word)  # Remove the leftover of zum / um zu
        return translation

    def conjugate(
        self,
        infinitive: str,
        tense: str,
        mood: str = "indicative",
        person: str = "er/sie/es",
    ) -> str:
        """Limited range of hardcoded rules (does NOT properly handle irregular verbs)"""

        # Enforce limitations
        if person != "er/sie/es":
            raise NotImplementedError("German conjugation only supports 3rd person singular")
        if mood != "indicative":
            raise NotImplementedError("German conjugation only supports indicative mood")

        # Check prefix
        infinitive = infinitive.replace("zu", "").strip()
        for prefix in self.separable_prefixes:
            if infinitive.startswith(prefix):
                stem = infinitive.replace(prefix, "")
                break
        else:
            stem = infinitive
            prefix = ""

        # Get the true stem by removing "-en" from infinitive
        stem = stem.replace("en", "")

        # Apply conjugation rules
        if tense == "präsens":
            return f"{stem}t {prefix}".strip()
        elif tense == "perfekt":
            # TODO-P3 handle auxiliary verb
            perfekt_ge = "" if any(stem.startswith(p) for p in self.inseparable_prefixes) else "ge"
            return f"{prefix}{perfekt_ge}{stem}t"
        elif tense == "imperfekt":
            return f"{stem}te {prefix}".strip()
        else:
            raise NotImplementedError(f"Unsupported tense {tense}")

    def extract_irregular_verb_forms(
        self,
        infinitive_en: str,
        infinitive_native: str,
        rely_on_google_translate: bool = False
    ) -> Optional[str]:

        # Irregulars are the same for all prefixes
        # TODO `voran` is a valid prefix (combining multiple)
        base_verb = self.prefix_regex.sub("", infinitive_native)
        irregular_key = next((k for k in (infinitive_native, base_verb) if k in self.cfg[self.IRREGULARS_KEY]), None)
        if irregular_key:
            return self.cfg[self.IRREGULARS_KEY][irregular_key]

        if not rely_on_google_translate:
            return None  # TODO need to wait for mlconjug3 to make this at all valuable

        # Get the reference conjugations in English
        kwargs = {
            "infinitive": infinitive_en,
            "mood": "indicative",
            "person": "he/she/it"
        }
        present = en.conjugate(tense="indicative present", **kwargs)
        simple_past = en.conjugate(tense="indicative past tense", **kwargs)
        perfect = en.conjugate(tense="indicative present perfect", **kwargs)

        # Convert to German
        # Add the explicit subject pronoun to avoid ambiguity
        # Simple past is rarely used in spoken German, so Google Translate has a preference for the perfekt
        # Since simple past is for narrating stories, adding "on the way" makes it think in this sense
        present_de = self.translate_from("he " + present, src="en").lower().replace("er ", "")
        simple_past_de = self.translate_from(
            "on the way, he " + simple_past, src="en"
        ).lower().replace(
            "er ", ""
        ).replace(
            "unterwegs", ""  # Remove prepositional phrase to leave just the verb
        ).strip()
        perfect_de = self.translate_from(
            "he had " + perfect, src="en"  # Force had to get perfekt construction
        ).lower().replace(
            "hatte", "hat"  # Google may think it's past perfekt
        ).replace(
            "war", "ist"
        ).split()[-1]  # Just interested in the gerund

        # Follow hardcoded conjugation to get the expected German (if verb was regular)
        present_expected = self.conjugate(infinitive_native, "präsens")
        simple_past_expected = self.conjugate(infinitive_native, "imperfekt")
        perfect_expected = self.conjugate(infinitive_native, "perfekt")

        # Compare to determine what's irregular
        simple_past_is_perfekt = "hat" in simple_past_de or "ist" in simple_past_de  # Ignore buggy Google translate
        tense_is_regular = [
            present_de == present_expected,
            simple_past_de == simple_past_expected or simple_past_is_perfekt,
            perfect_de == perfect_expected,  # Don't worry about sein vs. haben auxiliary verb
        ]
        note = ""
        for is_regular, conjugated in zip(tense_is_regular, [present_de, simple_past_de, perfect_de]):
            note += f"{conjugated if not is_regular else '_'}, "
        if not len(re.sub("[_,\\s]+", "", note)):
            note = None
        else:
            note = note[:-2]
        return note

    def extract_plural_ending(self, singular: str, plural: str) -> Optional[str]:

        # TODO reject ending if it's >3 (or 4?) characters. I don't think German plural endings get that long
        # TODO keep mapping of existing endings in memory to reference for compound nouns
        # Enforce basic rules
        singular_article, singular_noun = singular.split()
        if singular_article == "die":
            en_endings = ("ung", "heit", "keit", "tion", "sion")
            if any(singular_noun.endswith(e) for e in en_endings):
                return "-en"

        # Try to infer generic cases
        if len(plural.split()) != 2:
            print(f"Expected 2 words in German plural, got {plural}")
            return None
        article, plural_noun = plural.split()
        if article != "die":
            print(f"Plural German article should always be `die`, got {plural}")
            return None
        if singular_noun in plural_noun:
            ending = plural_noun.replace(singular_noun, "-")
        else:
            plural_noun_no_umlauts = self.remove_umlauts(plural_noun)
            if singular_noun in plural_noun_no_umlauts:
                ending = plural_noun_no_umlauts.replace(singular_noun, "")
                ending = "-̈" + ending
            else:
                print(f"Failed to find plural ending for {singular} -> {plural}")
                ending = None
        return ending

    def get_translation(self, word: Word) -> Optional[Word]:
        """Only nouns should be capitalized in German"""
        translation = super().get_translation(word)
        if translation and word.pos != PartOfSpeech.NOUN:
            translation.word = translation.word.lower()
        return translation

    def group_verbs(self, df: pd.DataFrame) -> Dict[str, List[Tuple[str, str]]]:
        """Group verbs after stripping prefixes and record definition for each prefix"""

        # Validate input
        required_cols = ("en", "de", "pos")
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Build prefix mapping
        base_verb_prefixes = defaultdict(list)
        for i, row in df.iterrows():
            if row.pos != PartOfSpeech.VERB or pd.isna(row.de):
                continue
            verb = row.de
            base_verb = self.prefix_regex.sub("", verb)
            if verb == base_verb:
                continue
            prefix = verb.replace(base_verb, "")
            base_verb_prefixes[base_verb].append((prefix, row.en.replace(" [v.]", "")))  # TODO use Word object

        # Add the base verb to the list of prefixes
        # Can't do this in the loop because the base verb may not have any prefixes
        for base_verb in list(base_verb_prefixes):
            matches = df.loc[df.de == base_verb]
            if matches.empty:
                continue  # Some verbs might use a prefix but their base form isn't in the list
            base_definition = ", ".join(en.replace(" [v.]", "") for en in matches.en)
            if len(matches) > 1:
                print(f"Multiple definitions for {base_verb}: {base_definition}")
            base_verb_prefixes[base_verb].append(("", base_definition))
        return base_verb_prefixes

    @property
    def prefix_regex(self) -> re.Pattern:
        """Regex to match any German verb prefix"""
        all_prefixes = "|".join(self.separable_prefixes + self.inseparable_prefixes)
        return re.compile(f"^({all_prefixes})")

    @staticmethod
    def remove_umlauts(de_text: str) -> str:
        replacements = {
            "ä": "a",
            "ö": "o",
            "ü": "u",
        }
        return "".join([replacements.get(c, c) for c in de_text])


en = English()
foreign_language_map = {
    "de": German(),
    "fr": French(),
}
