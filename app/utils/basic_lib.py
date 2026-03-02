import re
import unicodedata


def canonicalize_string(original: str) -> str:
    """
    Canonicalize a string using the same rules as kbScripts/basicLib.canonicalizeString.
    """
    s = unicodedata.normalize("NFKC", original)[:200]
    s = re.sub(r"[^\w\s-]", "", s.lower())
    s = re.sub(r"[-\s]+", "_", s).strip("-_")
    return s.replace("\t", "").replace("\n", "")
