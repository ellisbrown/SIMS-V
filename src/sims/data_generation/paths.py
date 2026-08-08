import os
from pathlib import Path


OBJAVERSE_SUBDIRECTORIES = ("processed", "houses", "procthor_databases")


def configured_objaverse_data_dir(*, cwd=None, environ=None):
    """Return the configured asset root without checking that it exists."""
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)
    candidate = environ.get("OBJAVERSE_DATA_DIR", cwd / "objaverse_sims")
    return Path(candidate).expanduser().resolve()


def resolve_objaverse_data_dir(
    explicit=None, *, required=False, cwd=None, environ=None
):
    """Resolve and export the Objaverse SIMS asset directory.

    Resolution order is an explicit CLI value, ``OBJAVERSE_DATA_DIR``, then an
    existing ``objaverse_sims`` directory under the current working directory.
    """
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)

    candidate = explicit or environ.get("OBJAVERSE_DATA_DIR")
    if candidate is None:
        conventional = cwd / "objaverse_sims"
        if conventional.is_dir():
            candidate = conventional

    if candidate is None:
        if required:
            raise FileNotFoundError(
                "Objaverse assets were not found. Download them to "
                "./objaverse_sims, pass --objaverse-dir, or set "
                "OBJAVERSE_DATA_DIR."
            )
        return None

    root = Path(candidate).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Objaverse asset directory does not exist: {root}")
    if required:
        missing = [
            name for name in OBJAVERSE_SUBDIRECTORIES if not (root / name).is_dir()
        ]
        if missing:
            joined = ", ".join(str(root / name) for name in missing)
            raise FileNotFoundError(
                f"Objaverse asset directory is incomplete; missing: {joined}"
            )

    environ["OBJAVERSE_DATA_DIR"] = str(root)
    return root
