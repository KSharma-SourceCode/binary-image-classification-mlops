import os
import re

VERSION_REGEX = re.compile(r"v(\d+)")

def get_next_version(base_path: str) -> str:
    """
    Returns next version as v1, v2, v3... based on existing folders.
    """
    if not os.path.exists(base_path):
        return "v1"

    versions = []
    for name in os.listdir(base_path):
        match = VERSION_REGEX.fullmatch(name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        return "v1"

    return f"v{max(versions) + 1}"


def get_latest_version(base_path: str) -> str:
    """
    Returns latest version folder (e.g., v3).
    """
    versions = []
    for name in os.listdir(base_path):
        match = VERSION_REGEX.fullmatch(name)
        if match:
            versions.append(int(match.group(1)))

    if not versions:
        raise ValueError(f"No versions found in {base_path}")

    return f"v{max(versions)}"
