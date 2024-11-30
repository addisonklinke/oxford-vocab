from argparse import ArgumentParser

from flashcard_builder.flashcards import FlashCardBuilder
from flashcard_builder.language import foreign_language_map


if __name__ == "__main__":

    parser = ArgumentParser(description="Convert Oxford PDF vocab list to CSV and add translations")
    parser.add_argument("-d", "--dst", type=str, required=True, help="Target language")
    parser.add_argument("-l", "--limit", type=int, help="Limit the number of rows translated")
    parser.add_argument("-p", "--pdf_path", type=str, required=True, help="Path to downloaded PDF")
    parser.add_argument("-s", "--split", type=str, choices=("level", "pos"), help="Save separate CSVs")
    parser.add_argument("--strict", action="store_true", help="Raise exceptions on translation failures")
    args = parser.parse_args()

    # TODO report percent success rate at each step (PDF to text, text to table, table to translation)
    # TODO option to load manual translations so edits can be preserved across changes to the code

    builder = FlashCardBuilder.from_oxford_pdf(
        pdf_path=args.pdf_path,
        dest=foreign_language_map[args.dst],
        limit=args.limit,
        strict=args.strict,
    )
    base_file = "".join(args.pdf_path.split(".")[:-1])
    builder.to_csv(base_file, split=args.split)
