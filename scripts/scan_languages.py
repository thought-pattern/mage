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
from os import getcwd as os_getcwd
from os import getenv as os_getenv
from os import makedirs as os_makedirs
from os import path as os_path
from os import walk as os_walk
from shutil import copy2 as shutil_copy2
from subprocess import PIPE as subprocess_PIPE
from subprocess import run as subprocess_run

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


def copy_language_files(rootfs: str, langfs: str):
    """
    Copy language files from the root filesystem to a language-specific
    directory structure to avoid scanning everything in the rootfs.
    """

    language_files = find_files(rootfs)

    for file in language_files:
        destination_dir = os_path.dirname(file).replace(rootfs, langfs)
        if not os_path.exists(destination_dir):
            os_makedirs(destination_dir)
        shutil_copy2(file, destination_dir)
    return False


def run_language_scan(langfs: str) -> str:
    """
    Scan the CVE database using the list of language packages found and save the
    results to a JSON file.

    Inputs
    =======
    langfs: str
        The directory containing the language package metadata files.

    Returns
    =======
    str
        The path to the JSON file containing the CVE scan results for the language packages.
        If the file does not exist, an empty string is returned.
    """

    print("Scanning Language Packages...")
    outfile = f"{CVE_DIR}/cve-bin-tool-lang-summary.json"

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
    return outfile


def main(rootfs: str) -> bool:
    """
    Scan the root filesystem for CVEs in the language packages.
    """
    copy_language_files(rootfs, f"{CVE_DIR}/langfs")
    run_language_scan(f"{CVE_DIR}/langfs")
    return False


if __name__ == "__main__":
    parser = argparse_ArgumentParser()
    parser.add_argument("rootfs", type=str)
    args = parser.parse_args()

    main(args.rootfs)
