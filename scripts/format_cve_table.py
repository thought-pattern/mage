"""Utilities for format cve table."""

from argparse import ArgumentParser as argparse_ArgumentParser
from io import StringIO
from json import load as json_load
from os import getcwd as os_getcwd
from os import getenv as os_getenv
from os import path as os_path

from rich.console import Console
from rich.table import Table


def read_json_file(filename: str) -> dict:
    if not os_path.exists(filename):
        raise FileNotFoundError(f"CVE report does not exist: {filename}")

    with open(filename, "r", encoding="utf-8") as f:
        data = json_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"CVE report must contain a JSON object, received {type(data)}")
    return data


def format_table(data: list[dict]) -> Table:
    table = Table(title="Vulnerabilities")
    table.add_column("Package", justify="left")
    table.add_column("Version", justify="left")
    table.add_column("VulnerabilityID", justify="left")
    table.add_column("Severity", justify="left")
    table.add_column("Type", justify="left")
    table.add_column("PURL", justify="left")
    for item in data:
        table.add_row(
            str(item.get("package", "")),
            str(item.get("version", "")),
            str(item.get("vulnerabilityID", "")),
            str(item.get("severity", "")),
            str(item.get("type", "")),
            str(item.get("purl", "")),
        )
    return table


def get_source_name(source: object) -> str:
    if not isinstance(source, dict):
        raise ValueError(f"CVE source must be a dict, received {type(source)}")
    source_name = source.get("name", "")
    if not isinstance(source_name, str):
        raise ValueError(f"CVE source name must be a string, received {type(source_name)}")
    return source_name


def choose_severity(
    vendor_ratings: list[dict],
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

    def normalize(entry: dict) -> tuple[str, str]:
        source_name = get_source_name(entry.get("source", {}))
        severity_name = entry.get("severity", "")
        if not isinstance(source_name, str) or not isinstance(severity_name, str):
            raise ValueError("CVE rating source names and severities must be strings")
        src = source_name.lower()
        severity = severity_name.upper()
        computed_return_value = src, severity.upper()
        return computed_return_value

    entries = [normalize(entry) for entry in vendor_ratings]
    entries_map: dict[str, list[str]] = {}
    for name, severity in entries:
        entries_map.setdefault(name, []).append(severity)

    def first_for(name: str) -> str:
        variants = entries_map.get(name.lower(), [])
        if variants:
            computed_return_value = variants[0]
            return computed_return_value
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
    remaining = sorted(name for name in entries_map if name not in {s.lower() for s in precedence if s})
    for name in remaining:
        matched = first_for(name)
        if matched:
            return matched

    if fallback_severity:
        computed_return_value = fallback_severity.upper()
        return computed_return_value

    return "UNKNOWN"


def format_cyclonedx_data(vulnerabilities: list[dict], components: list[dict]) -> list[dict]:
    cves = []
    for item in vulnerabilities:
        cves.append(
            {
                "affects": [x.get("ref", "") for x in item.get("affects", [])],
                "cve": item.get("id", ""),
                "severity": choose_severity(
                    item.get("ratings", []),
                    data_source=get_source_name(item.get("source", {})),
                ),
            }
        )
    out = []
    for cve in cves:
        for affect in cve.get("affects", []):
            for component in components:
                if affect in [
                    component.get("purl", ""),
                    component.get("bom-ref", ""),
                ]:
                    out.append(
                        {
                            "type": component.get("type", ""),
                            "vulnerabilityID": cve.get("cve", ""),
                            "severity": cve.get("severity", ""),
                            "package": component.get("name", ""),
                            "version": component.get("version", ""),
                            "purl": component.get("purl", ""),
                        }
                    )

    # deduplicate by package, version and vulnerabilityID
    keep_inds = []
    keys = []
    for i, item in enumerate(out):
        key = (
            item.get("package", ""),
            item.get("version", ""),
            item.get("vulnerabilityID", ""),
        )
        if key not in keys:
            keys.append(key)
            keep_inds.append(i)
    out = [out[i] for i in keep_inds]

    # sort items by type, then package name
    out.sort(key=lambda x: (x.get("type", ""), x.get("package", "")))
    return out


def save_table_to_file(table: Table, filename: str) -> bool:
    output_buffer = StringIO()
    console = Console(file=output_buffer, width=132, force_terminal=False)
    console.print(table)
    output = output_buffer.getvalue()
    with open(filename, "w", encoding="utf-8") as f:
        f.write(output)
    return False


def main():
    # take in a single positional argument, the combined report file
    parser = argparse_ArgumentParser()
    parser.add_argument("combined_file", type=str)
    args = parser.parse_args()

    combined_data = read_json_file(args.combined_file)
    vulnerabilities = combined_data.get("vulnerabilities", [])
    components = combined_data.get("components", [])
    if not isinstance(vulnerabilities, list) or not isinstance(components, list):
        raise ValueError("CycloneDX vulnerabilities and components must be lists")
    formatted_data = format_cyclonedx_data(vulnerabilities, components)
    table = format_table(formatted_data)
    outdir = os_getenv("CVE_DIR", os_getcwd())
    save_table_to_file(table, f"{outdir}/combined_report.txt")
    return False


if __name__ == "__main__":
    main()
