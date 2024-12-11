import re

SERIALIZED_WORD = re.compile(r"([a-z]+)(?:\s\((.+)\))? \[([a-z]+)\.\]")