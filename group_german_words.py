from argparse import ArgumentParser
from collections import defaultdict
from typing import Dict, List, Tuple

from flashcard_builder.io import concat_csvs
from flashcard_builder.language import German


def format_verbs(base_verb_prefixes: Dict[str, List[Tuple[str, str]]]) -> str:
    """Combine different definitions under the same prefix, sorted alphabetically, and add padding/indent"""
    output = ""
    for base_verb, prefixes in base_verb_prefixes.items():
        prefix_definitions = defaultdict(list)
        for prefix, definition in prefixes:
            prefix_definitions[prefix].append(definition)
        if len(prefix_definitions) == 1:
            continue
        output += "\n" + base_verb + "\n\t"
        pad = max(len(prefix) for prefix in prefixes) + len(base_verb)
        definitions = [
            f"{prefix + base_verb:<{pad}}: {', '.join(definitions)}"
            for prefix, definitions
            in sorted(prefix_definitions.items(), key=lambda x: x[0])
        ]
        output += "\n\t".join(definitions)
    return output


if __name__ == "__main__":

    parser = ArgumentParser(description="Group verbs by prefix and nouns by base word")
    parser.add_argument("file_glob", help="Quoted glob pattern for CSV files to concatenate")
    args = parser.parse_args()

    df = concat_csvs(args.file_glob)
    base_verb_prefixes = German().group_verbs(df)
    print(format_verbs(base_verb_prefixes).expandtabs(2))
