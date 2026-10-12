"""Reject experiment prompts that reveal the human comparison set."""

# Screened against the brief's lowercased text. "analysts typically" was
# removed 2026-08-26 (expabl Deviation 1): it false-positives on the
# standard-practice prose the brief prompt itself requests.
MARKERS = [
    "many-analyst",
    "many analyst",
    "many-team",
    "many teams, one dataset",
    "manyecoevo",
    "narps",
    "neuroimaging analysis replication",
    "multiverse",
    "analysts chose",
    "analysts in this study",
    "these analysts",
    "teams chose",
    "teams used",
    "most teams",
    "most analysts",
    "crowd",
    "forking paths",
    "multi-analyst",
    "silberzahn",
    "breznau",
    "fincap",
    "menkveld",
    "gould",
    "botvinik",
    "replication study",
    "analyst teams",
    "independent teams",
    "published result",
    "the study found",
    "originally reported",
    "fisher z",
    "fisher-z",
    "layer 1",
    "layer-1",
    "layer 2",
    "layer-2",
    "standardized effect",
]


def screen(text: str) -> list[str]:
    """Markers present in text, lowercased match; empty list means clean."""
    low = text.lower()
    return [marker for marker in MARKERS if marker in low]
