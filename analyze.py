from argparse import ArgumentParser
from collections import defaultdict
import re
from typing import Dict, List, Tuple

import pandas as pd

from flashcard_builder.io import concat_csvs
from flashcard_builder.language import German


def group_german_verbs(df: pd.DataFrame) -> None:

    def _format_verbs(base_verb_prefixes: Dict[str, List[Tuple[str, str]]]) -> str:
        """Combine different definitions under the same prefix, sorted alphabetically, and add padding/indent"""
        output = ""
        note_regex = re.compile(r" \[.*\]")
        for base_verb, prefixes in base_verb_prefixes.items():
            prefix_definitions = defaultdict(list)
            for prefix, definition in prefixes:
                prefix_definitions[prefix].append(definition)
            if len(prefix_definitions) == 1:
                continue
            output += "\n" + base_verb + "\n\t"  # TODO put the base definition on this line
            pad = max(len(prefix) for prefix in prefixes) + len(note_regex.sub("", base_verb))
            definitions = [
                f"{note_regex.sub('', prefix + base_verb):<{pad}}: {', '.join(definitions)}"
                for prefix, definitions
                in sorted(prefix_definitions.items(), key=lambda x: x[0])
            ]
            output += "\n\t".join(definitions)
        return output

    base_verb_prefixes = German().group_verbs(df)  # TODO get the language by dest column of CSV
    print(_format_verbs(base_verb_prefixes).expandtabs(2))


def list_ambiguous_translations(df: pd.DataFrame) -> None:
    """List different English words that received the same translation

    For flashcards in particular, the user may want to revise these with a
    more specific word
    """

    def _print_ambiguous_translations(group: pd.DataFrame) -> None:
        if len(group) == 1:
            return
        sep = "\n\t- "
        print(
            f"{group[dest].iloc[0]} assigned to {len(group)} English words:"
            f"{sep}{sep.join(word for word in group['en'])}".expandtabs(2)
        )

    expected_cols = ("en", "pos", "level")
    additional_cols = set(df.columns) - set(expected_cols)
    if len(additional_cols) == 0:
        raise ValueError(f"Expected a destination column in addition to {', '.join(expected_cols)}")
    if len(additional_cols) > 1:
        raise ValueError(f"Too many destination columns: {', '.join(additional_cols)}")
    dest = additional_cols.pop()
    print(f"Found {len(df.groupby(dest).filter(lambda group: len(group) > 1))} ambiguous translations")
    df.groupby(dest)[["en", dest]].apply(_print_ambiguous_translations)


if __name__ == "__main__":

    parser = ArgumentParser(description="Group verbs by prefix and nouns by base word")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser.add_argument("file_glob", help="Quoted glob pattern for CSV files to concatenate")
    german_verbs_subparser = subparsers.add_parser("german_verbs", help="Group German verbs by prefix")
    german_verbs_subparser.set_defaults(func=group_german_verbs)
    ambiguous_translations_subparser = subparsers.add_parser("ambiguous_translations", help="List ambiguous translations")
    ambiguous_translations_subparser.set_defaults(func=list_ambiguous_translations)
    args = parser.parse_args()

    df = concat_csvs(args.file_glob)
    args.func(df)
