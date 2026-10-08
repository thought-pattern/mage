"""
This script should speed up the scanning for specific programming-language
package vulnerabilities with cve-bin-tool.

This script doe the following:
1. Walks the extracted root filesystem of the container searching for package
metadata files which cve-bin-tool would normally look for (`valid_files`).

2. Copies these files to a separate directory structure so that cve-bin-tool
   does not have to scan the entire root filesystem, which can be very slow.

3. Runs cve-bin-tool on the copied files, using a triage file to filter out
   false positives, and outputs the results to a JSON file.
"""

from argparse import ArgumentParser as argparse_ArgumentParser
from json import JSONDecodeError as json_JSONDecodeError, load as json_load
from os import (
    getcwd as os_getcwd,
    getenv as os_getenv,
    makedirs as os_makedirs,
    path as os_path,
    replace as os_replace,
    walk as os_walk,
)
from shutil import copy2 as shutil_copy2
from subprocess import PIPE as subprocess_PIPE, run as subprocess_run
from tempfile import TemporaryDirectory

from cve_bin_tool.parsers.parse import valid_files as cbt_valid_files

CVE_DIR = os_getenv("CVE_DIR", os_getcwd())


def find_files(rootfs: str) -> list[str]:
    """
    Find all language files that CVE-bin-tool scans

    Inputs
    ======
    rootfs: str
        The root directory to search for language files

    Returns
    =======
    matches: list[str]
        A list of paths to metadata files for language packages
    """

    file_checkers = cbt_valid_files.copy()
    file_checkers["METADATA"] = file_checkers.get("METADATA: ", {})
    file_checkers["PKG-INFO"] = file_checkers.get("PKG-INFO: ", False)
    language_files = list(file_checkers.keys())

    matches = []
    for dirpath, _, filenames in os_walk(rootfs):
        for filename in filenames:
            if filename in language_files and filename != "requirements.txt":
                matches.append(f"{dirpath}/{filename}")
    return matches


def copy_language_files(rootfs: str, langfs: str) -> None:
    """
    Copy language files from the root filesystem to a language-specific
    directory structure to avoid scanning everything in the rootfs.
    """

    language_files = find_files(rootfs)

    os_makedirs(langfs)
    for file in language_files:
        destination_dir = os_path.join(langfs, os_path.relpath(os_path.dirname(file), rootfs))
        os_makedirs(destination_dir, exist_ok=True)
        shutil_copy2(file, destination_dir)


def run_language_scan(langfs: str, outfile: str) -> None:
    """
    Scan the CVE database using the list of language packages found and save the
    results to a JSON file.

    cve-bin-tool's exit status counts products with findings and overlaps its error codes, so completion
    is established by this attempt producing a complete JSON document at `outfile`, which the caller
    must give as a path no earlier run could have written.

    Inputs
    =======
    langfs: str
        The directory containing the language package metadata files.
    outfile: str
        The run-owned path for the JSON scan results.
    """

    print("Scanning Language Packages...")

    cmd = [
        "cve-bin-tool",
        "-u",
        "never",  # Never update the local CVE database
        "-f",
        "json",  # Output format: JSON
        "-o",
        outfile,  # Write JSON results to this file
        f"{langfs}/",
    ]
    completed = subprocess_run(cmd, stdout=subprocess_PIPE, stderr=subprocess_PIPE, text=True)
    if not os_path.isfile(outfile):
        raise RuntimeError(f"language CVE scan produced no result (status {completed.returncode}): {completed.stderr.strip()}")
    try:
        with open(outfile, "r", encoding="utf-8") as f:
            json_load(f)
    except (OSError, json_JSONDecodeError) as err:
        raise RuntimeError(f"language CVE scan result is incomplete (status {completed.returncode}): {err}") from err


def main(rootfs: str) -> None:
    """
    Scan the root filesystem for CVEs in the language packages.
    """
    # Each run stages its own metadata copy and scanner output, so files from an earlier run can be neither
    # mixed into this scan nor mistaken for its result; only a completed report replaces the published one.
    with TemporaryDirectory(dir=CVE_DIR, prefix="cve-bin-tool-lang-") as staging:
        staged_report = f"{staging}/cve-bin-tool-lang-summary.json"
        copy_language_files(rootfs, f"{staging}/langfs")
        run_language_scan(f"{staging}/langfs", staged_report)
        os_replace(staged_report, f"{CVE_DIR}/cve-bin-tool-lang-summary.json")


if __name__ == "__main__":
    parser = argparse_ArgumentParser()
    parser.add_argument("rootfs", type=str)
    args = parser.parse_args()

    main(args.rootfs)
