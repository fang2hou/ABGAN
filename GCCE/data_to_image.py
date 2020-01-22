import level_io as li
import numpy as np

def get_block_position(x, y, baseline):
    # 0 <= x, y, baseline <= 19
    x_offset = 0
    y_offset = -3.3
    block_size = .445

    x = x * block_size + x_offset
    y = (baseline - y) * block_size + y_offset

    return x, y

def convert(path):
    file_path = "../data/generated_data_wgan_gp_5_fz/from_net_01.txt"
    level_data = np.loadtxt(file_path, dtype=float, delimiter=',', encoding='utf8')
    level_data = np.reshape(level_data, (3, 20, 20))

    baseline = -1
    for y in range(19, -1, -1):
        for x in range(0, 20):
            if np.argmax(level_data[:, x, y]) != 0:
                baseline = y
                break
        if baseline != -1:
            break
    if baseline == -1:
        print("The input is blank.")
        exit(0)

    new_level = li.level()
    new_level.add_birds("BirdRed", 1)
    new_level.add_birds("BirdBlack", 4)
    new_level.add_birds("BirdBlue", 2)

    for x in range(0, 20):
        for y in range(0, 20):
            channel = np.argmax(level_data[:, x, y])
            if channel == 1:
                block_x, block_y = get_block_position(x, y, baseline)
                new_level.add_block(type="SquareSmall", x=block_x, y=block_y, rotation="0", material="ice")
            elif channel == 2:
                block_x, block_y = get_block_position(x, y, baseline)
                new_level.add_block(type="SquareSmall", x=block_x, y=block_y, rotation="0", material="wood")

    print('export')
    new_level.export(path)

if __name__ == "__main__":
    convert("level-4.xml")