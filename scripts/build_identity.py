"""MAGE and Memgraph build identity shared by the daily and PR build workflows."""

from json import load as json_load
from re import compile as re_compile
from subprocess import run as subprocess_run
from urllib import request as urllib_request


def extract_commit_hash(filename):
    """
    Attempts to extract a commit hash from the given filename.
    The regex looks for a delimiter, then 8 to 12 hexadecimal characters,
    followed by another delimiter.
    """
    # This regex looks for one of the delimiters (. _ + ~ -)
    # then captures a group of 8-12 hex digits,
    # and ensures it is followed by a delimiter like - or _ or .
    pattern = re_compile(r"[._+~-](?P<hash>[0-9a-f]{8,12})(?=[-_\.])")
    match = pattern.search(filename)
    if match:
        computed_return_value = match.group("hash")
        return computed_return_value
    return False


def get_commit():
    p = subprocess_run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"unable to resolve MAGE commit: {p.stderr.strip()}")

    computed_return_value = p.stdout.strip()
    return computed_return_value


def get_pr():
    p = subprocess_run(["git", "log", "--pretty=%B"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"unable to inspect MAGE history: {p.stderr.strip()}")
    pattern = re_compile(r"\(#(?P<pr>\d+)\)")
    match = pattern.search(p.stdout)
    computed_return_value = match.group("pr") if match else ""
    return computed_return_value


def get_tag():
    with urllib_request.urlopen("https://api.github.com/repos/memgraph/mage/tags") as response:
        # Read the JSON data from GitHub
        tags = json_load(response)

    # Find the first tag whose name does not contain 'rc'
    try:
        latest = next(tag.get("name", "")[1:] for tag in tags if "rc" not in tag.get("name", ""))
    except StopIteration as err:
        raise RuntimeError("GitHub returned no stable MAGE tag") from err
    return latest


def get_mage_version():
    commit = get_commit()
    pr = get_pr()
    tag = get_tag()

    mage_version = f"{tag}_pr{pr}_{commit}"
    return mage_version
