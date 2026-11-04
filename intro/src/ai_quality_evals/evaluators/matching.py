"""Transparent binary string matching rules, version 1.0.0."""

RULE_VERSION = "1.0.0"


def normalize(text: str) -> str:
    """Case-fold and collapse Unicode whitespace; preserve all other characters."""
    return " ".join(text.casefold().split())


def exact_match(candidate: str, reference: str) -> int:
    """Return 1 for unchanged string equality, otherwise 0."""
    return int(candidate == reference)


def normalized_match(candidate: str, reference: str) -> int:
    """Return 1 for equality under the documented v1 normalization rule."""
    return int(normalize(candidate) == normalize(reference))
