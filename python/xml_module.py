"""
Note: our implementation of xpath is different from neo4j,
because python offers different support for xpath than java
We cant have absolute paths
And our xpath search starts from the root node,
so ./something is equivalent to /root/something
For example,
python input : .//ZONE is equivalent to java input //ZONE
https://docs.python.org/3/library/xml.etree.elementtree.html#xpath-support
here are our avaliable xpath options
"""

from urllib import request as urllib_request

from defusedxml import ElementTree as ET
from mgp import Map as mgp_Map
from mgp import Record as mgp_Record
from mgp import function as mgp_function
from mgp import read_proc as mgp_read_proc

_DEFAULT_ARGUMENT_DICT = {}

TYPE = "_type"
TEXT = "_text"
CHILDREN = "_children"


def parse_element(element, simple):
    result = {TYPE: element.tag}

    attributes = element.attrib
    if attributes:
        result.update(attributes)

    text_content = element.text
    if text_content and text_content.strip():
        result[TEXT] = text_content

    children = list(element)
    if children:
        children_name = CHILDREN
        if simple:
            children_name = "_" + str(element.tag)
        result[children_name] = [parse_element(child, simple) for child in children]

    return result


def xml_file_to_string(xml_file):
    xml_string = ""
    try:
        with open(xml_file, "r") as xml_file:
            xml_string = xml_file.read().replace("\n", "").replace("  ", "")
    except PermissionError as _caught_error_53:
        raise PermissionError(
            "You don't have permissions to write into that file.Make sure to give the necessary permissions to user memgraph."
        ) from _caught_error_53
    except Exception as _caught_error_57:
        raise OSError("Could not open or write to file.") from _caught_error_57
    return xml_string


@mgp_function
def parse(xml_input: str, simple: bool = False, path: str = "") -> mgp_Map:
    """
    Function to parse xml string (or file) into a map.

    Parameters
    ----------
    xml_input : str
        XML string which is to be parsed.
    simple: bool = false
        Boolean which specifies how should the children list be named,
        when it is false, all children lists are named _children
        if true, all children lists are named based on their parent.
    path: str = ""
        Path to XML file which is to be parsed, if it is not "",
        XML string is ignored and only file is parsed, if it is left as
        default, it is ignored.

    Returns:
        mgp.Map -> XML file parsed as map
    """

    root = False
    parser = ET.DefusedXMLParser()
    if path:
        root = ET.fromstring(xml_file_to_string(path), parser)
    else:
        root = ET.fromstring(xml_input, parser)
    output_map = parse_element(root, simple)
    return output_map


def check_url(url):
    if not url.endswith(".xml"):
        raise ValueError("File must be xml!")
    return False


def xpath_search(root, xpath_expression):
    try:
        result = root.findall(xpath_expression)
        return result
    except Exception as e:
        raise ValueError(f"XPath search error: {e}") from e


@mgp_read_proc
def load(
    xml_url: str,
    simple: bool = False,
    path: str = "",
    xpath: str = "",
    headers: mgp_Map = _DEFAULT_ARGUMENT_DICT,
) -> mgp_Record(output_map=mgp_Map):
    """
    Procedure to load XML from url or from file to a map.

    Parameters
    ----------
    xml_url : str
        Url of the xml file to be parsed.
    simple: bool = false
        Boolean which specifies how should the children list be named,
        when it is false, all children lists are named _children
        if true, all children lists are named based on their parent.
    path: str = ""
        Path to XML file which is to be parsed, if it is not "",
        XML url is ignored and only file is parsed. If it is left as default,
        file path is ignored.
    xpath: str = ""
        Xpath expression which specifies which elements shall be returned.
        If left as "", it will be ignored, otherwise,
        only elements in XML file which satisfy
        expression are returned.
    headers: mgp.Map ={}
        Additional HTTP headers used in url request.

    Returns:
        mgp.Map -> XML file or URL parsed as map.
        In case XPATH is active, a map of for each element will be returned.
    """
    if headers is _DEFAULT_ARGUMENT_DICT:
        headers = _DEFAULT_ARGUMENT_DICT.copy()
    root = False
    parser = ET.DefusedXMLParser()
    if path:
        root = ET.fromstring(xml_file_to_string(path), parser)
    else:
        check_url(xml_url)
        try:
            request = urllib_request.Request(xml_url, headers=headers)
            response = urllib_request.urlopen(request).read()
            root = ET.fromstring(response)
        except Exception as e:
            raise ValueError(f"Error while fetching or parsing XML: {e}") from e

    if xpath:
        record_list = list()
        xpath_list = xpath_search(root, xpath)
        for element in xpath_list:
            record_list.append(mgp_Record(output_map=parse_element(element, simple)))
        return record_list
    output_map = parse_element(root, simple)
    _return_value = mgp_Record(output_map=output_map)
    return _return_value
