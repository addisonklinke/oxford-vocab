from argparse import ArgumentParser
from collections import defaultdict
from enum import StrEnum
from functools import partial
from itertools import combinations
import os
import re
import traceback
from typing import List, Optional
import warnings

import inflect
from googletrans import Translator
import mlconjug3
from mlconjug3 import Conjugator
from nltk import edit_distance
import pandas as pd
from pypdf import PdfReader
from sklearn.base import InconsistentVersionWarning
import spacy
import yaml

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)  # mlconjug3
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using")  # spacy/pytorch
conjugator = Conjugator(language="en")
translator = Translator()
try:
    nlp = spacy.load("en_core_web_trf")  # Recommended for best accuracy by https://spacy.io/models
except OSError as exc:
    raise RuntimeError("Missing spaCy model: run `python -m spacy download <model_name>` to fix") from exc
POS_CHARS = "nvadjcopredt"


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
        cfg_path = os.path.join(os.path.dirname(__file__), f"cfg/{self.name}.yaml")
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}
        for key in (self.SKIP_KEY, self.TRANSLATIONS_KEY, self.IRREGULARS_KEY, self.PLURALS_KEY):
            if (
                key not in self.cfg
                or self.cfg[key] is None
            ):
                self.cfg[key] = {}

        # Parse manually defined translations for disambiguating notes and POS
        # TODO consider allowing parenthetical note before English word (sometimes it reads more naturally that way)
        english_regex = re.compile(r"([a-z]+)(?:\s\((.+)\))? \[([a-z]+)\.\]")
        self.ambiguous_words = defaultdict(list)
        for english_full, translation in self.cfg[self.TRANSLATIONS_KEY].items():
            s = english_regex.match(english_full)
            if not s:
                print(f"Failed to parse manual translation: {english_full}")
                continue
            english_word, note, pos = s.groups()
            if note:
                self.ambiguous_words[(english_word, pos)].append((note, translation))

    def _get_noun_translation(self, english: str) -> str:
        # TODO handle bifurcated masculine/feminine nouns (i.e. [stem]erin, -nen)
        if self.noun_article:
            english = f"{self.noun_article} {english}"
        translation = en.translate_to(english, dest=self.name)
        if translation in self.cfg[self.PLURALS_KEY]:
            ending = self.cfg[self.PLURALS_KEY][translation]
        else:
            plural = en.translate_to(en.pluralize(english), dest=self.name)
            ending = self.extract_plural_ending(translation, plural)
        if ending:
            # TODO add distinction for plural only nouns (currently in YAML as ~)
            translation = translation + ", " + ending
        return translation

    def _get_verb_translation(self, english_infinitive: str) -> str:
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
        if note:
            translation = translation + " [" + note + "]"
        return translation

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
        english_word2level = df.set_index("en")["level"].to_dict()
        for i, row in df.iterrows():
            if (row.en, row.pos) in self.ambiguous_words:
                to_remove.append(i)
        df.drop(to_remove, axis=0, inplace=True)

        # Replace with manual translations
        new_rows = []
        for (english, pos), translations in self.ambiguous_words.items():
            if english not in english_word2level:
                continue  # Doesn't apply to this vocab list
            for note, translation in translations:
                new_rows.append({
                    "en": english,
                    self.name: translation,
                    "pos": pos,
                    "level": english_word2level[english],
                })
        assert len(new_rows) >= len(to_remove), "Failed to replace all ambiguous translations"
        return pd.concat([df, pd.DataFrame(new_rows)])

    def extract_irregular_verb_forms(self, infinitive_en: str, infinitive_native: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def extract_plural_ending(self, singular: str, plural: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def get_translation(self, english: str, pos: "PartOfSpeech") -> Optional[str]:
        """Translate a word from English into this language"""
        if f"{english} [{pos}.]" in self.cfg[self.SKIP_KEY]:
            return None
        manual_translation = self.cfg[self.TRANSLATIONS_KEY].get(english)
        if manual_translation:
            return manual_translation
        method_map = {
            PartOfSpeech.NOUN: self._get_noun_translation,
            PartOfSpeech.VERB: self._get_verb_translation,
        }
        method = method_map.get(pos, partial(self.translate_from, src="en"))
        # TODO return an object with word, context, POS, etc attrs and use a default formatter class to get the string
        translation = method(english)
        return re.sub(r"\s+", " ", translation)

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

    def get_translation(self, english: str, pos: "PartOfSpeech") -> str:
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

    def _get_noun_translation(self, english: str) -> str:
        """Make sure article is lowercase"""
        translation = super()._get_noun_translation(english)
        article, *noun = translation.split()
        article = article.lower()
        noun[0] = noun[0].capitalize()
        return article + " " + " ".join(noun)

    def _get_verb_translation(self, english_infinitive: str) -> str:
        translation = super()._get_verb_translation(english_infinitive)
        return re.sub("^u?m ", "", translation)  # Remove the leftover of zum / um zu

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
        all_prefixes = "|".join(self.separable_prefixes + self.inseparable_prefixes)
        base_verb = re.sub(f"^({all_prefixes})", "", infinitive_native)
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

        # TODO handle some basic plural rules like -ung and -heit
        # TODO reject ending if it's >3 (or 4?) characters. I don't think German plural endings get that long
        # TODO keep mapping of existing endings in memory to reference for compound nouns

        if len(plural.split()) != 2:
            print(f"Expected 2 words in German plural, got {plural}")
            return None
        article, plural_noun = plural.split()
        if article != "die":
            print(f"Plural German article should always be `die`, got {plural}")
            return None
        singular_noun = singular.split()[-1]
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

    def get_translation(self, english: str, pos: "PartOfSpeech") -> str:
        """Only nouns should be capitalized in German"""
        translation = super().get_translation(english, pos)
        if pos != PartOfSpeech.NOUN:
            return translation.lower()
        return translation

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


class OxfordPdf:
    """Parse Oxford vocab list PDFs into tabular format"""

    def __init__(self, pdf_path: str):
        self.lines = self.parse(pdf_path)

    @staticmethod
    def parse(pdf_path: str) -> List[str]:
        """Convert raw PDF text into word entries"""

        # Get the entire document as one string
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Preprocessing cleanup
        text = "".join([i for i in text if ord(i) < 128])  # Strip non-ASCII characters
        text = re.sub("(?<=[a-z])\n?[0-9]", "", text)  # Strip superscript that marauds as difficulty number
        text = text.replace("auxiliary", "").replace("modal", "")  # Italic annotations
        text = re.sub(rf"\(.+\)(\s+[{POS_CHARS}]+\.\s+[ABC][12])", r"\1", text)  # Clarifying notes in parentheses

        # Extract individual entries from the vocabulary list
        entry_regex = re.compile(
            "[a-zA-Z\\s]+\\s"  # English word (could contain space or be proper noun)
            f"[{POS_CHARS}"  # Letter abbreviations for POS
            "ACB12,.( )]+"  # Might have separate difficulty ratings for different POS
            "\\.,?\\s"  # But this middle section always ends in a period (and maybe a comma)
            "[ABC][12]"  # The last (typically only) difficulty rating
        )
        return re.findall(entry_regex, text)

    def to_df(self) -> pd.DataFrame:
        """Parse each entry string into a tabular row"""
        rows = []
        errors = 0
        ABBREV_REGEX = re.compile("[,.]")
        for line in self.lines:
            # TODO use regex/spaCy to detect unexpected POS in the English word (like punctuation, numbers, etc)
            parts = ABBREV_REGEX.sub("", line).split()  # Don't need comma and abbreviation periods anymore
            num_levels = len(re.findall("[ABC][12]", line))

            # Multiple difficulty ratings, implying multiple POS as well
            if num_levels > 1:
                pos_groups = re.findall(f"(?:(?:(?<=. )[{POS_CHARS}]+.(?:,\\s)?)+ [ABC][12])+", line)
                if len(pos_groups) != num_levels:
                    print(f"Expected one group of POS per level, got {pos_groups} for {num_levels} levels: {line}")
                    errors += 1
                    continue
                for group in pos_groups:
                    pos_parts = ABBREV_REGEX.sub("", group).split()
                    level = pos_parts[-1]
                    for pos in pos_parts[:-1]:
                        rows.append([
                            parts[0],
                            pos,
                            level,
                        ])

            elif line.count(".") > 1:
                # Multiple POS, all the same difficulty
                if len(parts) < 4:
                    print(f"Expected at least 4 parts, got {parts}")
                    errors += 1
                    continue
                poss = parts[1:-1]
                level = parts[-1]
                rows.extend([
                    [
                        parts[0],
                        pos,
                        level,
                    ]
                    for pos in poss
                ])
            else:
                # Standard format: one POS and one difficulty
                if len(parts) != 3:
                    print(f"Expected 3 parts, got {parts}")
                    errors += 1
                    continue
                word, pos, level = parts
                rows.append([
                    word,
                    pos,
                    level,
                ])
        if errors:
            print(
                f"Dataframe creation: {errors} errors on {len(self.lines)} lines"
                f" ({errors/len(self.lines) * 100:.2f}%)"
            )
        return pd.DataFrame(rows, columns=["en", "pos", "level"])


class PartOfSpeech(StrEnum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "adj"
    ADVERB = "adv"
    CONJUNCTION = "conj"
    PREPOSITION = "prep"
    DETERMINER = "det"


def dedupe(df: pd.DataFrame, dest: str, edit_dist_pct: float = 0.0) -> pd.DataFrame:
    """Remove duplicate vocab entries"""

    # Obvious ones where the same English word (under different POS) received the same translation
    before = len(df)
    df = df.drop_duplicates(subset=["en", dest])
    print(f"Removed {before - len(df)} exact duplicates")

    # Other times the translations might use different articles but are otherwise the same
    # Group by English and use edit distance to find similar translations
    # Calculate the average edit distance across all translations within a group
    # Then assign boolean column and use later for filtering
    # This is most often nouns and adjectives that are legitimate so filtering is off by default
    # a reasonable threshold seems to be 0.5 if you'd like to turn it on

    def _avg_edit_dist_within_thres(group) -> bool:
        if len(group) == 1:
            return False

        def _preproc(s):
            """Remove noun plurals/verb conjugations and switch to lowercase"""
            # TODO consider spaCy for language-specific logic to removing articles
            return re.sub(r"[,\\[].+$", "", s).lower()

        edit_distances = []
        for x, y in combinations(group, 2):
            if not x or not y:
                continue
            xp = _preproc(x)
            yp = _preproc(y)
            edit_distance_pct = edit_distance(xp, yp) / max(len(xp), len(yp))
            edit_distances.append(edit_distance_pct)
        if not edit_distances:
            return False
        avg_edit = sum(edit_distances) / len(edit_distances)
        return avg_edit < edit_dist_pct

    df["fuzzy_dupe"] = df.groupby("en")[dest].transform(_avg_edit_dist_within_thres)
    print(f"Removing {df['fuzzy_dupe'].sum()} fuzzy duplicates")
    df = df[~df["fuzzy_dupe"]].drop(columns=["fuzzy_dupe"])

    # There can also be different English words that received the same translation
    # These shouldn't be removed, but it's helpful to warn the user
    # For flashcards in particular, they may want to revise these with a more specific word

    def _print_ambiguous_translations(group: pd.DataFrame) -> None:
        if len(group) == 1:
            return
        sep = "\n\t- "
        print(
            f"{group[dest].iloc[0]} assigned to multiple English words:"
            f"{sep}{sep.join(group['en'])}".expandtabs(2)
        )

    print(f"Found {len(df.groupby(dest).filter(lambda group: len(group) > 1))} ambiguous translations")
    df.groupby(dest)[["en", dest]].apply(_print_ambiguous_translations)
    return df


def translate(
    pdf_path: str,
    language: Language,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Translate parsed output of PDF into destination language"""

    # Convert PDF into dataframe
    oxford = OxfordPdf(pdf_path)
    df = oxford.to_df()

    # Get translations
    translated = []
    total = len(df)
    for i, row in df.iterrows():
        if limit and i > limit:
            break
        try:
            translation = language.get_translation(row.en, row.pos)
        except KeyboardInterrupt:
            break
        except:
            print(f"Failed on row {i}: {row.en}")
            traceback.print_exc()
            translation = None
        translated.append(translation)
        print(f"Translated {(i + 1)/total * 100:.2f}%", end="\r")

    # Fill nulls when `limit` is set
    df[language.name] = translated + [None] * (len(df) - len(translated))

    # Disambiguate and/or clear out duplicates
    df = language.disambiguate(df)
    df = dedupe(df, language.name)

    # Include POS in English to disambiguate on flashcards
    df["en"] = df.apply(lambda row: f"{row.en} [{row.pos}.]", axis=1)

    # Reorder columns for easier copy-paste to flashcard services like Quizlet
    df = df[["en", language.name, "level", "pos"]]
    return df


if __name__ == "__main__":

    parser = ArgumentParser(description="Convert Oxford PDF vocab list to CSV and add translations")
    parser.add_argument("-d", "--dst", type=str, required=True, help="Target language")
    parser.add_argument("-l", "--limit", type=int, help="Limit the number of rows translated")
    parser.add_argument("-p", "--pdf_path", type=str, required=True, help="Path to downloaded PDF")
    parser.add_argument("-s", "--split", type=str, choices=("level", "pos"), help="Save separate CSVs")
    args = parser.parse_args()

    # TODO report percent success rate at each step (PDF to text, text to table, table to translation)
    # TODO option to load manual translations so edits can be preserved across changes to the code

    language = foreign_language_map[args.dst]
    output = translate(
        pdf_path=args.pdf_path,
        language=language,
        limit=args.limit,
    )
    base_file = "".join(args.pdf_path.split(".")[:-1])
    if args.split:
        for val in output[args.split].unique():
            new = output.loc[output[args.split] == val]
            split_path = args.dst + f"-{val}.csv"
            if os.path.isfile(split_path):
                existing = pd.read_csv(split_path)
                new = pd.concat([existing, new])
                new = dedupe(new, language.name)
            new.to_csv(split_path, index=False)
    else:
        output.to_csv(base_file + ".csv", index=False)
