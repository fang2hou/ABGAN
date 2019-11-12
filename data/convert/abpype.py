# Import libraries
import xml.etree.ElementTree as ET
import level_io as li
import numpy as np

WIDTH_MIN = -3.5
WIDTH_MAX = 8.5
HEIGHT_MIN = -3.5
HEIGHT_MAX = 5.0

TRAIN_SCALE_WIDTH = 128
TRAIN_SCALE_HEIGHT = 128


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
    data_x = round(TRAIN_SCALE_WIDTH * (x-WIDTH_MIN) / (WIDTH_MAX-WIDTH_MIN))
    data_y = round(TRAIN_SCALE_HEIGHT * (y-HEIGHT_MIN) / (HEIGHT_MAX-HEIGHT_MIN))
    return data_x, data_y

def inverse_normalize_postion(x, y):
    level_x = x / 128. * (WIDTH_MAX-WIDTH_MIN) + WIDTH_MIN
    level_y = y / 128. * (WIDTH_MAX-WIDTH_MIN) + WIDTH_MIN
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
                channel = names_to_channel[game_object.get('type')+str(rotation)] - 1
        else:
            channel = names_to_channel[tag]

        # Save
        index = (x-1) * TRAIN_SCALE_WIDTH + (y-1)
        data[channel][index] = 1

    # Output
    np.savetxt(out_path, data, delimiter=",", fmt='%1d', encoding='utf8')

def data_to_level(file_path, out_path):
    train_data = np.loadtxt(file_path, dtype=int, delimiter=',', encoding='utf8')


#if __name__ == "__main__":

data = np.loadtxt("../../model/results/from_net_1.gz", dtype=int, delimiter=',', encoding='utf8')
block_table = np.zeros((TRAIN_SCALE_HEIGHT, TRAIN_SCALE_WIDTH), dtype=int)

for x in range(TRAIN_SCALE_HEIGHT):
    for y in range(TRAIN_SCALE_WIDTH):
        index = x * TRAIN_SCALE_WIDTH + y
        channel = np.argmax(data[:, index])
        
        # If there is no block, skip it
        if data[channel, index] == 0:
            continue
        
        block_table[x][y] = channel + 1