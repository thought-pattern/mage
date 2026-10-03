"""Utilities for pr build vars."""

from argparse import ArgumentParser as argparse_ArgumentParser
from json import JSONDecodeError as json_JSONDecodeError
from json import loads as json_loads
from os import path as os_path
from subprocess import run as subprocess_run

from build_identity import extract_commit_hash, get_mage_version


def get_memgraph_version(pr):
    p = subprocess_run(
        [
            "aws",
            "s3",
            "ls",
            f"s3://deps.memgraph.io/pr-build/memgraph/pr{pr}/ubuntu-24.04-relwithdebinfo/",
            "--recursive",
        ],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"unable to list Memgraph PR build pr{pr}: {p.stderr.strip()}")

    # extract the file key found - there should only be one!
    files = [line.split()[3] for line in p.stdout.splitlines() if len(line.split()) > 3]
    if len(files) != 1:
        raise RuntimeError(f"expected one Memgraph PR build for pr{pr}, found {len(files)}")
    file = files[0]

    # remove the path, the first and last parts of the filename which should
    # always be the same to reveal the daily build version
    basename = os_path.basename(file)
    version = basename[9:-12]
    hash = extract_commit_hash(file)

    return version, hash


def pr_build_vars(payload):
    pr = payload.get("pr", False)
    if not isinstance(pr, str) or not pr:
        raise ValueError("PR build payload requires a non-empty pr string")
    if pr.startswith("pr"):
        pr = pr[2:]

    memgraph_version, memgraph_commit = get_memgraph_version(pr)

    mage_version = get_mage_version()

    return mage_version, memgraph_version, memgraph_commit, pr


def main() -> bool:
    parser = argparse_ArgumentParser(description="Read payload from Memgraph PR build workflow")

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

    mage_version, memgraph_version, memgraph_commit, pr = pr_build_vars(payload)
    print(f"{mage_version} {memgraph_version} {memgraph_commit} {pr}")
    return False


if __name__ == "__main__":
    main()
