"""Utilities for daily build vars."""

from argparse import ArgumentParser as argparse_ArgumentParser
from json import JSONDecodeError as json_JSONDecodeError
from json import load as json_load
from json import loads as json_loads
from os import path as os_path
from re import compile as re_compile
from subprocess import run as subprocess_run
from urllib import request as urllib_request


def get_latest_build() -> int:
    p = subprocess_run(
        [
            "aws",
            "s3",
            "ls",
            "s3://deps.memgraph.io/daily-build/memgraph/",
            "--recursive",
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"unable to list Memgraph daily builds: {p.stderr.strip()}")

    # extract the file keys found
    files = [line.split()[3] for line in p.stdout.splitlines()]

    # get the dates
    keydates = {file.split("/")[2] for file in files if len(file.split("/")) > 2}
    dates = [int(keydate) for keydate in keydates if keydate.isdecimal()]
    if not dates:
        raise RuntimeError("Memgraph daily-build listing contained no dated builds")

    computed_return_value = max(dates)
    return computed_return_value


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


def get_memgraph_version(date):
    p = subprocess_run(
        [
            "aws",
            "s3",
            "ls",
            f"s3://deps.memgraph.io/daily-build/memgraph/{date:08d}/ubuntu-24.04/",
            "--recursive",
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"unable to list Memgraph build {date:08d}: {p.stderr.strip()}")

    # extract the file key found - there should only be one!
    files = [line.split()[3] for line in p.stdout.splitlines() if len(line.split()) > 3]
    if len(files) != 1:
        raise RuntimeError(f"expected one Memgraph build for {date:08d}, found {len(files)}")
    file = files[0]

    # remove the path, the first and last parts of the filename which should
    # always be the same to reveal the daily build version
    basename = os_path.basename(file)
    version = basename[9:-12]
    hash = extract_commit_hash(file)

    return version, hash


def daily_build_vars(payload):
    if "date" in payload:
        date = payload.get("date", False)
    else:
        date = get_latest_build()

    memgraph_version, memgraph_commit = get_memgraph_version(date)

    mage_version = get_mage_version()

    return mage_version, memgraph_version, memgraph_commit, date


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


def main() -> bool:
    parser = argparse_ArgumentParser(description="Read payload from Memgraph daily build workflow")

    parser.add_argument(
        "payload",
        type=str,
        nargs="?",
        default="",
        help="JSON data from build workflow (optional)",
    )

    args = parser.parse_args()
    if args.payload:
        try:
            payload = json_loads(args.payload)
        except json_JSONDecodeError:
            payload = {}
    else:
        payload = {}

    mage_version, memgraph_version, memgraph_commit, date = daily_build_vars(payload)
    print(f"{mage_version} {memgraph_version} {memgraph_commit} {date}")
    return False


if __name__ == "__main__":
    main()
