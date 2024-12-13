import re

SERIALIZED_WORD = re.compile(r"([a-z\s]+)(?:\s\((.+)\))? \[([a-z]+)\.\]")
WHITESPACE_STRIP = re.compile(r"^\s+|\s+$|\s+(?=\s)")
