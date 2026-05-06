import xml.etree.ElementTree as ET


def parse_user_xml(xml_blob):
    root = ET.fromstring(xml_blob)
    return root
