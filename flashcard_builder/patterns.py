import re

SERIALIZED_WORD = re.compile(r"([a-z\s]+)(?:\s\((.+)\))? \[([a-z]+)\.\]")
