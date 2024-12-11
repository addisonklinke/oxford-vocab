from dataclasses import dataclass
from itertools import combinations
import os
import re
import traceback
from typing import Dict, List, Optional

from nltk import edit_distance
import pandas as pd

from .grammar import PartOfSpeech, Word
from .language import Language
from .sources.oxford_pdf import OxfordPdf


@dataclass
class Flashcard:

    front: Word
    back: Optional[Word]

    def to_row(self, front_col: str = "front", back_col: str = "back") -> Dict[str, str]:
        """Handle possibility of missing back translation"""
        return {
            front_col: self.front.format(),
            back_col: self.back.format(word_only=True) if self.back else None,
            "pos": self.front.pos,
            "level": self.front.level,
        }


@dataclass
class FlashcardSet:

    flashcards: List[Flashcard]

    def to_df(self, front_col: str = "front", back_col: str = "back") -> pd.DataFrame:  # TODO can dataframe type annotations contain columns?
        """Convert to DataFrame to take advantage of Pandas APIs for post-processing

        This maintains the series' elements as `Word` objects for better
        in-memory manipulation. To serialize this dataframe to CSV after
        post-processing, use `.write_csv()`
        """
        return pd.DataFrame(
            data=[(f.front, f.back) for f in self.flashcards],
            columns=[front_col, back_col]
        )

    @staticmethod
    def write_csv(df: pd.DataFrame, path: str) -> None:
        """Write DataFrame of `Word` objects to CSV with their notes and POS in string format"""
        assert len(df.columns) == 2, "DataFrame must have two columns (front and back)"


class FlashCardBuilder:

    def __init__(self, words: List[Word], dest: Language, limit: Optional[int] = None, strict: bool = False) -> None:
        self.words = words
        self.dest = dest
        self.limit = limit
        self.strict = strict

    def _dedupe(self, df: pd.DataFrame, dest: str, edit_dist_pct: float = 0.0) -> pd.DataFrame:
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

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.dest.disambiguate(df)
        df = self._dedupe(df, self.dest.name)
        df = df.dropna(subset=self.dest.name)
        return df

    def _translate(self, limit: Optional[int] = None) -> List[Word]:
        translated = []
        total = len(self.words)
        for i, word in enumerate(self.words):
            if limit and i > limit:
                break
            try:
                translation = self.dest.get_translation(word)
            except:
                if self.strict:
                    raise
                print(f"Failed on row {i}: {word.word}")
                traceback.print_exc()
                translation = None
            translated.append(translation)
            print(f"Translated {(i + 1) / total * 100:.2f}%", end="\r")

        # Fill nulls when `limit` is set
        translated += [None] * (len(self.words) - len(translated))
        return translated

    def to_csv(self, base_file: str, split: Optional[str] = None) -> None:
        flashcard_set = self.build()
        df = flashcard_set.to_df(front_col="en", back_col=self.dest.name)
        df = self._postprocess(df)
        if split:
            assert split in df.columns, f"Split column {split} not found in DataFrame"
            for val in df[split].unique():
                new = df.loc[df[split] == val]
                split_path = self.dest.name + f"-{val}.csv"
                if os.path.isfile(split_path):
                    existing = pd.read_csv(split_path)
                    new = pd.concat([existing, new])
                    new = self._dedupe(new, self.dest.name)
                new.to_csv(split_path, index=False)
        else:
            df.to_csv(base_file + ".csv", index=False)

    def build(self) -> FlashcardSet:
        translations = self._translate(self.limit)
        assert len(translations) == len(self.words), "Mismatch between words and translations"
        return FlashcardSet([
            Flashcard(front=word, back=translation)
            for word, translation in zip(self.words, translations)
        ])

    @classmethod
    def from_src(cls, words_src: str, *args, **kwargs) -> "FlashCardBuilder":
        """Detect source of words and dispatch to appropriate constructor"""
        if words_src.endswith(".pdf"):
            return cls.from_oxford_pdf(words_src, *args, **kwargs)
        raise NotImplementedError(f"Unsupported source: {words_src}")

    @classmethod
    def from_oxford_pdf(cls, pdf_path: str, *args, **kwargs) -> "FlashCardBuilder":
        oxford = OxfordPdf(pdf_path)
        df = oxford.to_df()  # TODO change this to Word list
        words = []
        for i, row in df.iterrows():
            words.append(Word(word=row.en, pos=PartOfSpeech(row.pos), level=row.level))
        return cls(words, *args, **kwargs)
