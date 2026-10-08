"""Utilities for daily build vars."""

from argparse import ArgumentParser as argparse_ArgumentParser
from json import JSONDecodeError as json_JSONDecodeError, loads as json_loads
from os import path as os_path
from subprocess import run as subprocess_run

from build_identity import extract_commit_hash, get_mage_version


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
