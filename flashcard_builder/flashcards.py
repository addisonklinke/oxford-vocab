from dataclasses import dataclass, fields
import os
import traceback
from typing import List, Optional

import pandas as pd

from .grammar import PartOfSpeech, Word
from .language import Language
from .sources.oxford_pdf import OxfordPdf


@dataclass
class Flashcard:

    front: Word
    back: Optional[Word]


@dataclass
class FlashcardSet:

    flashcards: List[Flashcard]
    src: str
    dest: str

    @property
    def serialized_columns(self) -> List[str]:
        return [self.src, self.dest, "pos", "level"]

    def to_word_df(self) -> pd.DataFrame:  # TODO can dataframe type annotations contain columns?
        """Wrap Word objects in Pandas DataFrame to take advantage of post-processing APIs

        This maintains the series' elements as `Word` objects for better
        in-memory manipulation. To serialize this dataframe to CSV after
        post-processing, use `.write_csv()`
        """
        return pd.DataFrame(
            data=[(f.front, f.back) for f in self.flashcards],
            columns=[self.src, self.dest]
        )

    def to_serializable_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame of `Word` objects for CSV with their notes and POS in string format"""

        # Get level and POS before converting Word objects (use front side)
        df["pos"] = df[self.src].apply(lambda word: word.pos.value)
        df["level"] = df[self.src].apply(lambda word: word.level)

        # Serialize Word objects to strings (differently for front vs. back)
        df[self.src] = df[self.src].apply(lambda word: word.format_front())
        df[self.dest] = df[self.dest].apply(lambda word: word.format_back())
        return df


class FlashCardBuilder:

    def __init__(self, words: List[Word], dest: Language, limit: Optional[int] = None, strict: bool = False) -> None:
        self.words = words
        self.dest = dest
        self.limit = limit
        self.strict = strict

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["en", self.dest.name])
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

        # Maintain `Word` objects throughout post-processing for better in-memory manipulation
        flashcard_set = self.build()
        df = flashcard_set.to_word_df()
        df = self.dest.disambiguate(df)  # Doesn't need to happen when loading/combining splits
        df = self._clean(df)

        # Only switch to string format for final serialization
        col_order = flashcard_set.serialized_columns
        if split:
            word_fields = {field.name for field in fields(Word)}
            if split not in word_fields:
                raise ValueError(f"Split column {split} must be a Word field")
            df[split] = df["en"].apply(lambda word: getattr(word, split, None))
            for val in df[split].unique():
                filtered = df.loc[df[split] == val]
                df_ser = flashcard_set.to_serializable_df(filtered)
                split_csv = self.dest.name + f"-{val}.csv"  # Shared name to combine multiple sources
                split_path = os.path.join(os.path.dirname(base_file), split_csv)
                if os.path.isfile(split_path):
                    existing = pd.read_csv(split_path)
                    df_ser = pd.concat([existing, df_ser])
                    df_ser = self._clean(df_ser)
                df_ser.to_csv(split_path, index=False, columns=col_order)
        else:
            df_ser = flashcard_set.to_serializable_df(df)
            df_ser.to_csv(base_file + ".csv", index=False, columns=col_order)

    def build(self) -> FlashcardSet:
        translations = self._translate(self.limit)
        assert len(translations) == len(self.words), "Mismatch between words and translations"
        return FlashcardSet(
            flashcards=[
                Flashcard(front=word, back=translation)
                for word, translation in zip(self.words, translations)
            ],
            src="en",
            dest=self.dest.name,
        )

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
