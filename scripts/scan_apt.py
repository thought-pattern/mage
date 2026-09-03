"""
This script is used for scanning a Docker container's installed APT packages, as
part of an effort to speed up the usage of `cve-bin-tool`.

This script does the following:
1. Executes a command within the container to list all installed packages in a
JSON format. This format only includes the `product` and `version`. We also need
`vendor`, which is less easy to fetch.

2. It then imports the CVE database object directly from cve_bin_tool to map
each of the installed packages to a `vendor` (or multiple in some cases).

3. Saves a CSV with columns: vendor, product, version.

4. Calls cve-bin-tool with the `--input-file` (`-i`) argument pointing to the
CSV file. This will do a direct database lookup for each product, vendor and
version, rather than scanning the iamge itself. The output is a JSON file
containing all CVEs for those installed packages.
"""

from argparse import ArgumentParser as argparse_ArgumentParser
from json import loads as json_loads
from os import getcwd as os_getcwd
from os import getenv as os_getenv
from subprocess import PIPE as subprocess_PIPE
from subprocess import run as subprocess_run

from cve_bin_tool.cvedb import CVEDB

CVE_DIR = os_getenv("CVE_DIR", os_getcwd())


def get_apt_packages(container: str = "memgraph") -> list[dict]:
    """
    Collect the list of installed apt packages within the container.

    Inputs
    ======
    container: str
       Name of the container to scan

    Returns
    =======
    packages: list of installed packages
    """

    cmd = [
        "docker",
        "exec",
        container,
        "dpkg-query",
        "--show",
        '--showformat={"name": "${binary:Package}", "version": "${Version}"}, ',
    ]

    result = subprocess_run(
        cmd,
        stdout=subprocess_PIPE,
        stderr=subprocess_PIPE,
        text=True,  # so that stdout/stderr come back as Python strings
    )
    if result.returncode != 0:
        raise RuntimeError(f"docker package query failed with exit code {result.returncode}: {result.stderr}")

    output = result.stdout.strip()
    if not output:
        return []

    # Remove trailing comma and wrap in array brackets
    output = output.rstrip(", ")
    packages = json_loads(f"[{output}]")

    print(f"Found {len(packages)} installed DEB packages")
    return packages


def get_package_vendor_pairs(cve_db: CVEDB, packages: list[dict]) -> list[dict]:
    """
    return the list of vendors for a package

    Inputs
    ======
    cve_db: CVEDB
      CVEDB object to use
    packages: List[str]
      list of installed packages

    Returns
    =======
    pairs: list of vendor/product/version dicts
    """

    computed_return_value = cve_db.get_vendor_product_pairs(packages)
    return computed_return_value


def combine_vendor_product_version(packages: list[dict], pairs: list[dict]) -> list[tuple[str]]:
    """
    create the full list of vendor, product and version for each package

    Inputs
    ======
    packages: List[str]
      list of installed packages
    pairs: List[str]
      list of vendor/product dicts

    Returns
    =======
    out: List[Tuple[str]]
        list of tuples (vendor, product, version)
    """
    prod_vends = {}
    for pair in pairs:
        prod = pair.get("product", False)
        vend = pair.get("vendor", False)
        if prod not in prod_vends:
            prod_vends[prod] = []
        prod_vends.get(prod, []).append(vend)

    out = []
    for package in packages:
        prod = package.get("name", "")
        ver = package.get("version", "")

        if prod in prod_vends:
            vends = prod_vends.get(prod, [])
            for vend in vends:
                out.append((vend, prod, ver))

    return out


def save_apt_package_csv(packages):
    """
    Save CSV of package vendors, products and versions

    Inputs:
      packages: List[str]
        list of installed packages
    """

    with open(f"{CVE_DIR}/apt-packages.csv", "w") as f:
        f.write("vendor,product,version\n")
        for package in packages:
            f.write(f"{','.join(package)}\n")
    return False


def run_scan() -> bool:
    """
    Run scan of apt packages and save results to JSON file.
    """

    cmd = [
        "cve-bin-tool",
        "-u",
        "never",  # Never update the local CVE database
        "-f",
        "json",  # Output format: JSON
        "-o",
        f"{CVE_DIR}/cve-bin-tool-apt-summary.json",  # Write JSON results to this file
        "-i",
        f"{CVE_DIR}/apt-packages.csv",
    ]
    result = subprocess_run(cmd, stdout=subprocess_PIPE, stderr=subprocess_PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cve-bin-tool failed with exit code {result.returncode}: {result.stderr}")
    return False


def main(container):
    """
    Scan container packages for CVEs

    Inputs
    ======
    container: str
        container name or ID
    """
    cve_db = CVEDB()

    packages = get_apt_packages(container)
    pairs = get_package_vendor_pairs(cve_db, packages)
    package_info = combine_vendor_product_version(packages, pairs)
    save_apt_package_csv(package_info)
    print(f"Checking {len(package_info)} packages with cve-bin-tool...")
    run_scan()
    del cve_db
    return False


if __name__ == "__main__":
    parser = argparse_ArgumentParser()
    parser.add_argument("container", type=str)
    args = parser.parse_args()

    main(args.container)
