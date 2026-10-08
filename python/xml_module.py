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

from io import DEFAULT_BUFFER_SIZE
from time import monotonic as time_monotonic
from urllib import request as urllib_request

from defusedxml import ElementTree as ET
from mgp import Map as mgp_Map, Record as mgp_Record, function as mgp_function, read_proc as mgp_read_proc

DEFAULT_ARGUMENT_DICT = {}

# A URL source is remote and unsized until it is read. These bounds own its acquisition: the whole fetch must finish
# within the deadline (each blocking socket operation also times out at it), and the body may not exceed the byte
# allowance. Parsing and output-map construction are linear in the admitted document, so the allowance bounds them too.
XML_URL_TIMEOUT_SECONDS = 60.0
XML_URL_MAX_BYTES = 64 * 1024 * 1024

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


def read_xml_file(xml_file: str) -> bytes:
    """
    Returns the file's original bytes so the XML parser sees exactly the document the string branch would: the
    parser honors the declared encoding, and significant whitespace in text and attributes is preserved.
    Whitespace-only presentation text is dropped afterwards by parse_element.
    """
    try:
        with open(xml_file, "rb") as xml_stream:
            xml_bytes = xml_stream.read()
    except PermissionError as err:
        raise PermissionError(
            "You don't have permissions to read that file. Make sure to give the necessary permissions to user memgraph."
        ) from err
    except OSError as err:
        raise OSError(f"Could not open or read XML file {xml_file}: {err}") from err
    return xml_bytes


def fetch_xml_bytes(xml_url: str, headers: dict) -> bytes:
    """
    Fetches an XML document within XML_URL_TIMEOUT_SECONDS and XML_URL_MAX_BYTES, closing the response on every
    path. read1 performs at most one blocking read per call, so the deadline is checked between bounded reads.
    """
    request = urllib_request.Request(xml_url, headers=headers)
    deadline = time_monotonic() + XML_URL_TIMEOUT_SECONDS
    chunks: list[bytes] = []
    received_bytes = 0
    try:
        with urllib_request.urlopen(request, timeout=XML_URL_TIMEOUT_SECONDS) as response:
            while True:
                chunk = response.read1(DEFAULT_BUFFER_SIZE)
                if not chunk:
                    break
                received_bytes += len(chunk)
                if received_bytes > XML_URL_MAX_BYTES:
                    raise ValueError(f"XML response from {xml_url} exceeds {XML_URL_MAX_BYTES} bytes")
                if time_monotonic() > deadline:
                    raise TimeoutError(f"XML response did not complete within {XML_URL_TIMEOUT_SECONDS} seconds")
                chunks.append(chunk)
    except OSError as err:
        raise ValueError(f"Error while fetching XML from {xml_url}: {err}") from err
    xml_bytes = b"".join(chunks)
    return xml_bytes


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
    if path:
        root = ET.fromstring(read_xml_file(path))
    else:
        root = ET.fromstring(xml_input)
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
    except Exception as err:
        raise ValueError(f"XPath search error: {err}") from err


@mgp_read_proc
def load(
    xml_url: str,
    simple: bool = False,
    path: str = "",
    xpath: str = "",
    headers: mgp_Map = DEFAULT_ARGUMENT_DICT,
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
    if headers is DEFAULT_ARGUMENT_DICT:
        headers = DEFAULT_ARGUMENT_DICT.copy()
    root = False
    if path:
        root = ET.fromstring(read_xml_file(path))
    else:
        check_url(xml_url)
        xml_bytes = fetch_xml_bytes(xml_url, headers)
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as err:
            raise ValueError(f"Error while parsing XML from {xml_url}: {err}") from err

    if xpath:
        record_list = list()
        xpath_list = xpath_search(root, xpath)
        for element in xpath_list:
            record_list.append(mgp_Record(output_map=parse_element(element, simple)))
        return record_list
    output_map = parse_element(root, simple)
    computed_return_value = [mgp_Record(output_map=output_map)]
    return computed_return_value
