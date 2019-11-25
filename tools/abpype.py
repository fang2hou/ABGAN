# Import libraries
import xml.etree.ElementTree as ET
from lxml import etree as ETL
import xml.dom.minidom as MD
import os
import numpy as np

WIDTH_MIN = 0
WIDTH_MAX = 8.5
HEIGHT_MIN = -3.5
HEIGHT_MAX = 5.0

TRAIN_SCALE_WIDTH = 128
TRAIN_SCALE_HEIGHT = 128

# From level_io
# https://github.com/fang2hou/Science-Birds-Edit-for-GCCE-2019
# Easy-to-go class for building Science Birds Levels.


class level():
    def __init__(self):
        # root
        root = ETL.Element('Level')
        # subelements
        ETL.SubElement(root, 'Camera', {
            # default camera
            "x": "0",
            "y": "-1",
            "minWidth": "15",
            "maxWidth": "17.5",
        })
        ETL.SubElement(root, 'Birds')
        ETL.SubElement(root, 'Slingshot', {
            "x": "-5",
            "y": "-2.5",
        })
        ETL.SubElement(root, 'GameObjects')

        self.root = root

    def add_birds(self, type, num):
        birds_node = self.root.find('Birds')
        for _ in range(0, num):
            ETL.SubElement(birds_node, "Bird", {"type": type})

    def add_block(self, type, x, y, rotation=0, material=None):
        block_node = self.root.find('GameObjects')

        if type == "TNT":
            ETL.SubElement(block_node, "TNT", {
                "x": str(x),
                "y": str(y),
                "rotation": str(rotation),
            })
            return True
        elif type == "Platform":
            ETL.SubElement(block_node, "Platform", {
                "type": "Platform",
                "x": str(x),
                "y": str(y),
                "rotation": str(rotation),
            })
            return True
        elif type == "Pig":
            ETL.SubElement(block_node, "Platform", {
                "type": "BasicSmall",
                "x": str(x),
                "y": str(y),
                "material": "",
                "rotation": str(rotation),
            })
            return True
        elif material is not None:
            ETL.SubElement(block_node, "Block", {
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
        ETL.ElementTree(self.root).write(filename, xml_declaration=True,
                                         encoding='utf-16', method='xml', pretty_print=True)


# Channel name
channel_to_names = {
    1: 'SquareHole', 2: 'RectFat', 3: 'RectFat90', 4: 'SquareSmall',
    5: 'SquareTiny', 6: 'RectTiny', 7: 'RectTiny90', 8: 'RectSmall',
    9: 'RectSmall90', 10: 'RectMedium', 11: 'RectMedium90',
    12: 'RectBig', 13: 'RectBig90', 14: 'TNT', 15: 'Pig', 16: 'Platform',
    17: 'TriangleHole', 18: 'Triangle', 19: 'Triangle90', 20: 'Circle',
    21: 'CircleSmall'
}

names_to_channel = dict([(n, c) for (c, n) in channel_to_names.items()])

# Normalization


def normalize_position(x, y):
    data_x = round(TRAIN_SCALE_WIDTH * (x - WIDTH_MIN) / (WIDTH_MAX - WIDTH_MIN))
    data_y = round(TRAIN_SCALE_HEIGHT * (y - HEIGHT_MIN) / (HEIGHT_MAX - HEIGHT_MIN))
    return data_x, data_y


def inverse_normalize_postion(x, y):
    level_x = x / 128. * (WIDTH_MAX - WIDTH_MIN) + WIDTH_MIN
    level_y = y / 128. * (WIDTH_MAX - WIDTH_MIN) + WIDTH_MIN
    return level_x, level_y

# Convert the level to input data


def level_to_data(file_path, out_path):
    # Initialize the array
    data = np.zeros(
        (len(channel_to_names), TRAIN_SCALE_WIDTH * TRAIN_SCALE_HEIGHT), dtype=int)

    # Parsing
    root = ET.parse(file_path).getroot()
    game_objects = root.find('GameObjects')

    for game_object in game_objects:
        tag = game_object.tag
        x = float(game_object.get('x'))
        y = float(game_object.get('y'))

        # Convert the position to the data scale
        x, y = normalize_position(x, y)

        # Find channel which is belongs to
        if tag == 'Block':
            channel = names_to_channel[game_object.get('type')] - 1
            rotation = int(float(game_object.get('rotation')))
            # Set the ratated block as another block type
            if rotation != 0:
                channel = names_to_channel[game_object.get('type') + str(rotation)] - 1
        else:
            channel = names_to_channel[tag]

        # Save
        index = (x - 1) * TRAIN_SCALE_WIDTH + (y - 1)
        data[channel][index] = 1

    # Output
    np.savetxt(out_path, data, delimiter=",", fmt='%1d', encoding='utf8')


def data_to_level(file_path, out_path, threshold=0.5):
    data = np.loadtxt(file_path, dtype=float, delimiter=',', encoding='utf8')
    block_table = np.zeros((TRAIN_SCALE_HEIGHT, TRAIN_SCALE_WIDTH), dtype=int)

    # Preview the structure
    for x in range(TRAIN_SCALE_HEIGHT):
        for y in range(TRAIN_SCALE_WIDTH):
            index = x * TRAIN_SCALE_WIDTH + y
            channel = np.argmax(data[:, index])

            # If there is no block, skip it
            if data[channel, index] < threshold:
                continue

            block_table[x][y] = channel + 1

    # TODO:Tricks

    # Initialize the level
    new_level = level()
    new_level.add_birds("BirdRed", 1)
    new_level.add_birds("BirdBlack", 4)
    new_level.add_birds("BirdBlue", 2)

    for x in range(TRAIN_SCALE_HEIGHT):
        for y in range(TRAIN_SCALE_WIDTH):
            real_x, real_y = inverse_normalize_postion(x + 1, y + 1)

            if block_table[x][y] == 0:
                continue

            block_name = channel_to_names[block_table[x][y]]

            normal_blocks = [
                'SquareHole', 'RectFat', 'SquareSmall', 'SquareTiny', 'RectTiny',
                'RectSmall', 'RectSmall', 'RectMedium', 'RectBig',
                'TriangleHole', 'Triangle', 'Circle', 'CircleSmall',
                'TNT', 'Pig', 'Platform',
            ]

            rotate_blocks = [
                'RectFat90', 'RectTiny90', 'RectSmall90', 'RectMedium90', 'RectBig90',
                'Triangle90'
            ]

            if block_name in normal_blocks:
                new_level.add_block(type=block_name, x=real_x, y=real_y, material="ice")
            elif block_name in rotate_blocks:
                new_level.add_block(type=block_name[:-2], x=real_x, y=real_y, rotation=90, material="ice")
            else:
                print('ERROR: Create {} block failed.'.format(block_name))

    new_level.export(out_path)
