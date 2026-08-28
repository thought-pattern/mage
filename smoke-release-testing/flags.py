"""Utilities for flags."""

from subprocess import PIPE as subprocess_PIPE
from subprocess import run as subprocess_run
from sys import argv as sys_argv
from xml.etree import ElementTree as ET


def extract_flags(to_run):
    ret = {}
    data = subprocess_run(to_run, stdout=subprocess_PIPE).stdout.decode("utf-8")
    data = "\n".join([line for line in data.split("\n") if line.startswith("<")])
    root = ET.fromstring(data)
    for child in root:
        if child.tag == "usage" and child.text.lower().count("warning"):
            raise Exception("You should set the usage message!")
        if child.tag == "flag":
            flag = {}
            for elem in child:
                flag[elem.tag] = elem.text if elem.text is not None else ""
            flag["override"] = False
            ret[flag.get("name", "")] = flag
    return ret


# The intention below was just to print flags into a file and make diff to
# manually observer the difference. Future extenstion is to take the same
# values and run memgraph between versions with non-default flags.
memgraph_docker_image = sys_argv[1]
flags = extract_flags(
    ["docker", "run", "-it", "--rm", memgraph_docker_image, "--help-xml"]
)
for flag_name, flag_details in flags.items():
    print(flag_name, flag_details.get("default", False))
# IMPORTANT: There is also an issue with printing default values, because
# --help is returning build defaults which is missleaning to the user but also
# here because we can't detect changes in default values.
