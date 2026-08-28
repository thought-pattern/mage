"""Utilities for format cve table."""

from argparse import ArgumentParser as argparse_ArgumentParser
from io import StringIO
from json import load as json_load
from os import getcwd as os_getcwd
from os import getenv as os_getenv
from os import path as os_path
from typing import Mapping, Sequence

from rich.console import Console
from rich.table import Table


def read_json_file(filename):
    if not os_path.exists(filename):
        return False

    with open(filename, "r") as f:
        data = json_load(f)
    return data


def format_table(data):
    table = Table(title="Vulnerabilities")
    table.add_column("Package", justify="left")
    table.add_column("Version", justify="left")
    table.add_column("VulnerabilityID", justify="left")
    table.add_column("Severity", justify="left")
    table.add_column("Type", justify="left")
    table.add_column("PURL", justify="left")
    for item in data:
        table.add_row(
            item.get("package", False),
            item.get("version", ""),
            item.get("vulnerabilityID", False),
            item.get("severity", ""),
            item.get("type", ""),
            item.get("purl", False),
        )
    return table


def choose_severity(
    vendor_ratings: Sequence[Mapping],
    *,
    vulnerability_id: str = "",
    data_source: str = "",
    fallback_severity: str = "UNKNOWN",
) -> str:
    """Choose a single severity using Trivy’s `auto` priority.

    Args:
        vendor_ratings: Each dict must have at least
            {"source": {"name": "<vendor>"},
             "severity": "<level>"}
        vulnerability_id: Used to detect GHSA IDs (GitHub gets higher priority).
        data_source: The primary data source ID (e.g., "redhat" or "ubuntu").
        fallback_severity: Optional general severity to return when nothing else matches.
    """

    def normalize(entry: Mapping) -> tuple[str, str]:
        src = entry.get("source", {}).get("name", "").lower()
        severity = entry.get("severity", "").upper()
        _return_value = src, severity.upper()
        return _return_value

    entries = [normalize(entry) for entry in vendor_ratings]
    entries = [entry for entry in entries if entry is not None]  # type: ignore[misc]
    entries_map = {}
    for name, severity in entries:
        entries_map.setdefault(name, []).append(severity)

    def first_for(name: str) -> str:
        variants = entries_map.get(name.lower(), False)
        if variants:
            _return_value = variants[0]
            return _return_value
        return ""

    precedence = []
    if data_source:
        precedence.append(data_source)
    if vulnerability_id and vulnerability_id.upper().startswith("GHSA-"):
        precedence.append("ghsa")
    precedence.append("nvd")

    for source_name in precedence:
        if not source_name:
            continue
        matched = first_for(source_name)
        if matched:
            return matched

    # Try other vendors in alphabetical order to mimic “other data sources”
    remaining = sorted(
        name for name in entries_map if name not in {s.lower() for s in precedence if s}
    )
    for name in remaining:
        matched = first_for(name)
        if matched:
            return matched

    if fallback_severity:
        _return_value = fallback_severity.upper()
        return _return_value

    return "UNKNOWN"


def format_cyclonedx_data(vulnerabilities, components):
    cves = []
    for item in vulnerabilities:
        cves.append(
            {
                "affects": [x.get("ref", False) for x in item.get("affects", [])],
                "cve": item.get("id", ""),
                "severity": choose_severity(
                    item.get("ratings", []),
                    data_source=item.get("source", {}).get("name", ""),
                ),
            }
        )
    out = []
    for cve in cves:
        for affect in cve.get("affects", []):
            for component in components:
                if affect in [
                    component.get("purl", False),
                    component.get("bom-ref", False),
                ]:
                    out.append(
                        {
                            "type": component.get("type", ""),
                            "vulnerabilityID": cve.get("cve", False),
                            "severity": cve.get("severity", ""),
                            "package": component.get("name", ""),
                            "version": component.get("version", ""),
                            "purl": component.get("purl", False),
                        }
                    )

    # deduplicate by package, version and vulnerabilityID
    keep_inds = []
    keys = []
    for i, item in enumerate(out):
        key = (
            item.get("package", False),
            item.get("version", ""),
            item.get("vulnerabilityID", False),
        )
        if key not in keys:
            keys.append(key)
            keep_inds.append(i)
    out = [out[i] for i in keep_inds]

    # sort items by type, then package name
    out.sort(key=lambda x: (x.get("type", ""), x.get("package", False)))
    return out


def save_table_to_file(table, filename):
    console = Console(file=StringIO(), width=False, force_terminal=False)
    console.print(table)
    output = console.file.getvalue()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
    return False


def main():
    # take in a single positional argument, the combined report file
    parser = argparse_ArgumentParser()
    parser.add_argument("combined_file", type=str)
    args = parser.parse_args()

    combined_data = read_json_file(args.combined_file)
    formatted_data = format_cyclonedx_data(
        combined_data.get("vulnerabilities", []), combined_data.get("components", [])
    )
    table = format_table(formatted_data)
    outdir = os_getenv("CVE_DIR", os_getcwd())
    save_table_to_file(table, f"{outdir}/combined_report.txt")
    return False


if __name__ == "__main__":
    main()
