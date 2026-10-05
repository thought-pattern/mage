"""Utilities for get cve image url."""

from argparse import ArgumentParser as argparse_ArgumentParser
from os import getenv as os_getenv

from aggregate_build_tests import list_daily_release_packages


def main(arch: str, image_type: str) -> bool:
    """
    print the relevant image URL to be scanned for CVEs

    Inputs
    ======
    arch: str
        the architecture of the image to be scanned for CVEs
    image_type: str
        the type of image to be scanned for CVEs, either 'memgraph' or 'mage'

    """
    date_str = os_getenv("CURRENT_BUILD_DATE")
    if not date_str:
        raise ValueError("CURRENT_BUILD_DATE environment variable is required")
    date = int(date_str)

    # translate to the daily-build package-map keys
    if image_type == "memgraph":
        os_key = "docker"
        arch_key = "arm64" if arch == "arm64" else "x86_64"
    elif image_type == "mage":
        os_key, arch_key = ("Docker (arm64)", "arm64") if arch == "arm64" else ("Docker (x86_64)", "x86_64")
    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

    packages = list_daily_release_packages(date, image_type=image_type)
    os_packages = packages.get(os_key, {})
    if not isinstance(os_packages, dict):
        raise RuntimeError(f"daily build {date} package entry for {os_key} must be a mapping")
    # A missing build fails the scan; another image is never substituted for the requested one.
    url = os_packages.get(arch_key, "")
    if not url:
        raise RuntimeError(f"daily build {date} has no {image_type} image for {os_key} {arch_key}")

    print(url)
    return False


if __name__ == "__main__":
    parser = argparse_ArgumentParser()
    parser.add_argument("arch", type=str)
    parser.add_argument(
        "image_type",
        type=str,
        choices=["memgraph", "mage"],
        help="type of image to use",
    )
    args = parser.parse_args()

    main(args.arch, args.image_type)
