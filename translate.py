from argparse import ArgumentParser
from enum import StrEnum
from functools import partial
import re
import string
from typing import List, Optional
import warnings

import inflect
from googletrans import Translator
import mlconjug3
from mlconjug3 import Conjugator
import pandas as pd
from pypdf import PdfReader
from sklearn.base import InconsistentVersionWarning


warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
conjugator = Conjugator(language="en")
translator = Translator()


class Language:

    name: str
    noun_article: Optional[str] = "the"

    def conjugate(
        self,
        infinitive: str,
        mood: str,
        tense: str,
        person: str
    ) -> str:
        """Return verb's conjugation in this language"""
        if self.name not in mlconjug3.LANGUAGES:
            # TODO revisit after https://github.com/Ars-Linguistica/mlconjug3/issues/331
            raise NotImplementedError(f"Language {self.name} not supported by mlconjug3")
        verb = conjugator.conjugate(infinitive)
        return verb[mood][tense][person]  # May raise KeyError for invalid parameters

    def _get_noun_translation(self, english: str) -> str:
        # TODO handle bifurcated masculine/feminine nouns (i.e. [stem]erin, -nen)
        if self.noun_article:
            english = f"{self.noun_article} {english}"
        translation = en.translate_to(english, dest=self.name)
        plural = en.translate_to(en.pluralize(english), dest=self.name)
        ending = self.extract_plural_ending(translation, plural)
        if ending:
            translation = translation + ", " + ending
        return translation

    def _get_verb_translation(self, english_infinitive: str) -> str:
        translated_infinitive_prefix = en.translate_to("to", dest=self.name)
        translation = en.translate_to(
            text="to " + english_infinitive,  # Use infinitive to ensure ambiguous words aren't treated as nouns
            dest=self.name
        )
        translation = translation.replace(translated_infinitive_prefix, "")
        note = self.extract_irregular_verb_forms(english_infinitive, translation)
        if note:
            translation = translation + "[" + note + "]"
        return translation

    def extract_irregular_verb_forms(self, infinitive_en: str, infinitive_native: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def extract_plural_ending(self, singular: str, plural: str) -> Optional[str]:
        """Subclasses can define language specific behavior. None tells consumers to ignore"""
        return None

    def get_translation(self, english: str, pos: "PartOfSpeech") -> str:
        """Translate a word from English into this language"""
        method_map = {
            PartOfSpeech.NOUN: self._get_noun_translation,
            PartOfSpeech.VERB: self._get_verb_translation,
        }
        method = method_map.get(pos, partial(self.translate_from, src="en"))
        return method(english)

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
    name = "en"

    def conjugate(
        self,
        infinitive: str,
        mood: str = "indicative",
        tense: str = "present",
        person: str = "he/she/it"
    ) -> str:
        """Set default to present tense 3rd person singular"""
        return super().conjugate(infinitive, mood, tense, person)

    def get_translation(self, english: str, pos: "PartOfSpeech") -> str:
        return english


class French(Language):
    noun_article = "a"  # Better for getting genders


class German(Language):

    name = "de"

    # https://coffeebreaklanguages.com/2024/06/making-sense-of-german-separable-verbs-a-guide-for-learners/
    separable_prefixes = (
        "ab",
        "an",
        "auf",
        "aus",
        "ein",
        "mit",
        "nach",
        "vor",
        "zu",
    )
    inseparable_prefixes = (
        "be",
        "emp",
        "ent",
        "er",
        "ver",
        "zer",
    )

    def extract_irregular_verb_forms(self, infinitive_en: str, infinitive_native: str) -> Optional[str]:

        # Get the reference conjugations in English
        _translate = partial(en.translate_to, dest="de")
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
        present_de = _translate("he " + present).lower().replace("er ", "")
        simple_past_de = _translate("he " + simple_past).lower().replace(
            "er ", "")  # FIXME returning perfekt instead of simple past
        perfect_de = _translate(
            "he had " + perfect  # Force had to get perfekt construction
        ).lower().replace(
            "hatte", "hat"  # Google may think it's past perfekt
        ).replace(
            "war", "ist"
        ).split()[-1]  # Just interested in the gerund

        # Determine whether they're irregular
        for prefix in self.separable_prefixes:
            if infinitive_native.startswith(prefix):
                stem = infinitive_native.replace(prefix, "")
                break
        else:
            stem = infinitive_native
            prefix = ""

        # TODO override `conjugate` method with this basic logic
        perfekt_ge = "" if any(stem.startswith(p) for p in self.inseparable_prefixes) else "ge"
        stem = stem.replace("en", "")
        present_expected = f"{stem}t {prefix}".strip()
        simple_past_expected = f"{stem}te {prefix}".strip()
        perfect_expected = f"{prefix}{perfekt_ge}{stem}t"
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
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        text = "".join([i for i in text if ord(i) < 128])  # Strip non-ASCII characters
        text = re.sub("(?<=[a-z])\n[0-9]", "", text)  # Strip superscript that marauds as difficulty number
        entry_regex = re.compile(
            # TODO handle proper nouns like month names
            "[a-z\\s]+\\s"  # English word is always lowercase
            "[nvadjco"  # Letter abbreviations for POS
            "ACB12,.\\s]+"  # Might have separate difficulty ratings for different POS
            "\\.\\s"  # But this middle section always ends in a period 
            "[ABC][12]"  # The last (typically only) difficulty rating
        )
        return re.findall(entry_regex, text)

    def to_df(self) -> pd.DataFrame:
        """Parse each entry string into a tabular row"""
        rows = []
        for line in self.lines:
            # TODO handle `modal` and `auxiliary` annotations in italics
            # TODO conditional lowercase depending on language (i.e. non-nouns in German) and proper nouns
            parts = re.sub("[,.]", "", line).split()  # Don't need comma and abbreviation periods anymore
            if len(re.findall(f"[{string.ascii_uppercase}]", line)) > 1:
                # Multiple difficulty ratings, implying multiple POS as well
                if len(parts) != 5:
                    print(f"Expected 5 parts, got {parts}")
                    continue
                word, pos1, level1, pos2, level2 = parts
                rows.extend([
                    [
                        word,
                        pos1,
                        level1,
                    ],
                    [
                        word,
                        pos2,
                        level2,
                    ]
                ])
            elif line.count(".") > 1:
                # Multiple POS, all the same difficulty
                if len(parts) < 4:
                    print(f"Expected at least 4 parts, got {parts}")
                    continue
                word = parts[0]
                poss = parts[1:-1]
                level = parts[-1]
                rows.extend([
                    [
                        word,
                        pos,
                        level,
                    ]
                    for pos in poss
                ])
            else:
                # Standard format: one POS and one difficulty
                if len(parts) != 3:
                    print(f"Expected 3 parts, got {parts}")
                    continue
                word, pos, level = parts
                rows.append([
                    word,
                    pos,
                    level,
                ])
        return pd.DataFrame(rows, columns=["en", "pos", "level"])


class PartOfSpeech(StrEnum):
    NOUN = "n"
    VERB = "v"
    ADJECTIVE = "adj"
    ADVERB = "adv"
    CONJUNCTION = "conj"


def translate(
    pdf_path: str,
    language: Language,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Translate parsed output of PDF into destination language"""
    oxford = OxfordPdf(pdf_path)
    df = oxford.to_df()
    translated = []
    total = len(df)
    for i, row in df.iterrows():
        if limit and i > limit:
            break
        try:
            translation = language.get_translation(row.en, row.pos)
        except Exception as e:
            print(f"Failed on row {i}: {row.en} -> {repr(e)}")
            translation = None
        translated.append(translation)
        print(f"Translated {(i + 1)/total * 100:.2f}%", end="\r")
    # TODO dedupe words, `fast` definitely appears in A1 list several times
    #  due to multiple POS. It's okay if they have different translations
    #  but in this case everything is `schnell` so it's a waste
    #  Also cardinal directions are 4x repeat, but articles are slightly different so they'd need to be grouped
    #  Probably something like df.groupby("en").dedupe(...) and look for x% overlapping characters
    df[language.name] = translated + [None] * (len(df) - len(translated))
    df["en"] = df.apply(lambda row: f"{row.en} [{row.pos}.]", axis=1)

    # TODO maybe helpful to print words that are different in English but received the same translation?
    #  Example: carry and wear both map to tragen
    return df


if __name__ == "__main__":

    parser = ArgumentParser(description="Convert Oxford PDF vocab list to CSV and add translations")
    parser.add_argument("-d", "--dst", type=str, required=True, help="Target language")
    parser.add_argument("-l", "--limit", type=int, help="Limit the number of rows translated")
    parser.add_argument("-p", "--pdf_path", type=str, required=True, help="Path to downloaded PDF")
    parser.add_argument("-s", "--split", type=str, choices=("level", "pos"), help="Save separate CSVs")
    args = parser.parse_args()

    # TODO report percent success rate at each step (PDF to text, text to table, table to translation)
    language = foreign_language_map[args.dst]
    output = translate(
        pdf_path=args.pdf_path,
        language=language,
        limit=args.limit,
    )
    base_file = "".join(args.pdf_path.split(".")[:-1])
    if args.split:
        for val in output[args.split].unique():
            output.loc[output[args.split] == val].to_csv(base_file + f"-{val}.csv", index=False)
    else:
        output.to_csv(base_file + ".csv", index=False)
