import re
from typing import List, Dict, Optional

# Known acronyms that should be preserved in uppercase
_ACRONYMS = {"RT", "RW", "DKI", "DIY"}

# Roman numerals that should stay uppercase
_ROMAN_NUMERALS = {
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"
}


def _titlecase_with_acronyms(text: str) -> str:
    """Convert to title case but keep known acronyms in uppercase."""
    return " ".join(
        word.upper() if word.upper() in _ACRONYMS
        else word.capitalize()
        for word in text.split()
    )


def _titlecase_with_romans(text: str) -> str:
    """Titlecase text while preserving roman numerals in uppercase."""
    return " ".join(
        word.upper() if word.upper() in _ROMAN_NUMERALS else word.capitalize()
        for word in text.split()
    )


def _extract_number(text: str) -> Optional[str]:
    """Extract first sequence of digits from text."""
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else None


def _extract_postalcode(text: str) -> Optional[str]:
    """Extract 5-digit postal code from text."""
    match = re.search(r"\b\d{5}\b", text)
    return match.group(0) if match else None


def normalize_components(ents: List[Dict]) -> Dict[str, Optional[str]]:
    """
    Normalize named entity components into structured address parts.

    Args:
        ents: List of entity dicts, each containing keys like 'entity' or 'entity_group',
              and 'text' or 'word'.

    Returns:
        Dict with normalized address components.
    """
    components = {
        "province": None,
        "city": None,
        "district": None,
        "village": None,
        "street": None,
        "rt": None,
        "rw": None,
        "postalcode": None,
    }

    for ent in ents:
        label = ent.get("entity_group") or ent.get("entity")
        text = (ent.get("word") or ent.get("text") or "").strip(" ,.;")

        if not text or not label:
            continue

        if label == "PROVINCE":
            components["province"] = _titlecase_with_acronyms(text)

        elif label == "CITY":
            components["city"] = _titlecase_with_acronyms(text)

        elif label == "DISTRICT":
            components["district"] = _titlecase_with_romans(text)

        elif label == "VILLAGE":
            components["village"] = _titlecase_with_romans(text)

        elif label in {"STREET", "BLOCK", "HAMLET"}:
            street_part = _titlecase_with_romans(text)
            components["street"] = (
                f"{components['street']} {street_part}".strip()
                if components["street"] else street_part
            )

        elif label == "RT":
            components["rt"] = _extract_number(text)

        elif label == "RW":
            components["rw"] = _extract_number(text)

        elif label == "POSTALCODE":
            components["postalcode"] = _extract_postalcode(text)

    return components
