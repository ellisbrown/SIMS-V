import os

import prior

# Cache for lazy loading
_objaverse_annotations = None


def get_objaverse_annotations():
    """Lazy-load and cache Objaverse annotations."""
    global _objaverse_annotations
    if _objaverse_annotations is None:
        _objaverse_annotations = prior.load_dataset(
            "objaverse-plus",
            entity="ellisbrown",
            revision="1bd4b77de24e76849e627af8e248437c6748e346",
            offline=os.environ.get("PRIOR_OFFLINE", "0") == "1",
        )["train"].data
    return _objaverse_annotations


def __getattr__(name):
    """Module-level attribute access for lazy loading."""
    if name == "OBJAVERSE_ANNOTATIONS":
        return get_objaverse_annotations()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
