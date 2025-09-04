import re
from typing import List, Dict

# simple titlecase but preserve known acronyms
_ACRONYMS = {"RT", "RW", "DKI", "DIY"}

def titlecase_keep(s: str) -> str:
    parts = s.split()
    out = []
    for p in parts:
        if p.upper() in _ACRONYMS:
            out.append(p.upper())
        else:
            out.append(p[:1].upper() + p[1:].lower() if p else p)
    return " ".join(out)

def normalize_components(ents: List[Dict]) -> Dict:
    out = {
        "province": None,
        "city": None,
        "district": None,
        "village": None,
        "street": None,
        "rt": None,
        "rw": None,
        "postalcode": None
    }
    for e in ents:
        label = e.get("entity_group") or e.get("entity")  # support different pipeline outputs
        text = e.get("word") or e.get("text") or ""
        text = text.strip(" ,.;")

        if label == "PROVINCE":
            out["province"] = titlecase_keep(text)

        elif label == "CITY":
            out["city"] = titlecase_keep(text)

        elif label == "DISTRICT":
            words = text.split()
            new_words = []
            for word in words:
                if word.upper() in {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                                    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"}:
                    new_words.append(word.upper())
                else:
                    new_words.append(word.capitalize())
            out["district"] = " ".join(new_words)

        elif label == "VILLAGE":
            words = text.split()
            new_words = []
            for word in words:
                if word.upper() in {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                                    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX"}:
                    new_words.append(word.upper())
                else:
                    new_words.append(word.capitalize())
            out["village"] = " ".join(new_words)


        elif label in {"STREET", "BLOCK", "HAMLET"}:
            text = titlecase_keep(text)
            
            if out["street"]:
                out["street"] += " " + text
            else:
                out["street"] = text

        elif label == "RT":
            m = re.search(r"(\d+)", text)
            if m:
                out["rt"] = m.group(1)  

        elif label == "RW":
            m = re.search(r"(\d+)", text)
            if m:
                out["rw"] = m.group(1) 


        elif label == "POSTALCODE":
            m = re.search(r"\d{5}", text)
            if m:
                out["postalcode"] = m.group(0)

    return out
