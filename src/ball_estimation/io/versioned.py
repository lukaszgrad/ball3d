import logging
import os

import pandas as pd

_logger = logging.getLogger(__name__)


def load_versioned_df(
    directory: str,
    base_name: str,
    version: str | None = None,
    extensions: tuple[str, ...] = (".feather", ".csv"),
) -> pd.DataFrame:
    """Load a DataFrame trying versioned then unversioned filenames.

    Resolution order (e.g. version="nieDL", base_name="ball_detection"):
      1. ball_detection.nieDL.feather
      2. ball_detection.nieDL.csv
      3. ball_detection.feather   (unversioned fallback)
      4. ball_detection.csv

    Raises
    ------
    FileNotFoundError
        If no matching file is found.
    """
    candidates: list[str] = []
    if version is not None:
        for ext in extensions:
            candidates.append(os.path.join(directory, f"{base_name}.{version}{ext}"))
    for ext in extensions:
        candidates.append(os.path.join(directory, f"{base_name}{ext}"))

    for path in candidates:
        if os.path.isfile(path):
            _logger.info("Loading %s", path)
            if path.endswith(".feather"):
                return pd.read_feather(path)
            return pd.read_csv(path, low_memory=False)

    raise FileNotFoundError(
        f"No file found for base='{base_name}', version='{version}' "
        f"in {directory}. Tried: {candidates}"
    )


def versioned_path(
    directory: str,
    base_name: str,
    version: str | None = None,
    extension: str = ".csv",
) -> str:
    """Build an output path with dot-separated version convention.

    Examples
    --------
    >>> versioned_path("/data/track", "ball_3d", "nieDL")
    '/data/track/ball_3d.nieDL.csv'
    >>> versioned_path("/data/track", "ball_3d")
    '/data/track/ball_3d.csv'
    """
    if version is not None:
        filename = f"{base_name}.{version}{extension}"
    else:
        filename = f"{base_name}{extension}"
    return os.path.join(directory, filename)
