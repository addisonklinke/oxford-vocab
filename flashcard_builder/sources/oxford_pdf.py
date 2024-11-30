import re
from typing import List

import pandas as pd
from pypdf import PdfReader

from ..grammar import POS_CHARS


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
