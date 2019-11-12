from lxml import etree as ET
import xml.dom.minidom as MD
import os

class level():
    def __init__(self):
        # root
        root = ET.Element('Level')
        # subelements
        ET.SubElement(root, 'Camera', {
            # default camera
            "x":"0",
            "y":"-1",
            "minWidth":"15",
            "maxWidth":"17.5",
        })
        ET.SubElement(root, 'Birds')
        ET.SubElement(root, 'Slingshot', {
            "x":"-5",
            "y":"-2.5",
        })
        ET.SubElement(root, 'GameObjects')

        self.root = root

    def add_birds(self, type, num):
        birds_node = self.root.find('Birds')
        for _ in range(0, num):
            ET.SubElement(birds_node, "Bird", {"type": type})

    def add_block(self, type, x, y, rotation=0, material=None):
        block_node = self.root.find('GameObjects')

        if type == "TNT":
            ET.SubElement(block_node, "TNT", {
                "x": str(x),
                "y": str(y),
                "rotation": str(rotation),
            })
            return True
        elif type == "Platform":
            ET.SubElement(block_node, "Platform", {
                "type": "Platform",
                "x": str(x),
                "y": str(y),
                "rotation": str(rotation),
            })
            return True
        elif type == "Pig":
            ET.SubElement(block_node, "Platform", {
                "type": "BasicSmall",
                "x": str(x),
                "y": str(y),
                "material":"",
                "rotation": str(rotation),
            })
            return True
        elif material is not None:
            ET.SubElement(block_node, "Block", {
                "type": str(type),
                "material": str(material),
                "x": str(x),
                "y": str(y),
                "rotation": str(rotation),
            })
            return True
        
        return False

    def export(self, filename):
        root = self.root
        ET.ElementTree(self.root).write(filename, xml_declaration=True, encoding='utf-16', method='xml', pretty_print=True)