from PIL import Image
import numpy as np


def get_block_type(pixel):
    brightness = pixel

    if brightness < 64:
        return None
    else:
        return "SquareSmall"


def get_block_position(x, y, baseline):
    # 0 <= x, y, baseline <= 19
    x_offset = 0
    y_offset = -3.3
    block_size = .445

    x = x * block_size + x_offset
    y = (baseline - y) * block_size + y_offset

    return x, y


def generate(pic_data, out_path):
    paint = Image.fromarray(pic_data)
    paint = paint.resize((20, 20))
    pix_data = np.asarray(paint)

    data_none = np.ones((20, 20), dtype=int)
    data_ice = np.zeros((20, 20), dtype=int)
    data_wood = np.zeros((20, 20), dtype=int)

    # -----------------
    # find baseline
    baseline = -1
    for y in range(19, -1, -1):
        for x in range(0, 20):
            if get_block_type(pix_data[x, y]) is not None:
                baseline = y
                break
        if baseline != -1:
            break
    if baseline == -1:
        print("The input is blank.")
        exit(0)
    # -----------------

    for x in range(0, 20):
        # Confirm the column has blocks
        has_blocks = False
        for y in range(baseline, -1, -1):
            if get_block_type(pix_data[x, y]) is not None:
                has_blocks = True
                final_block = y
        # ----------------------
        if has_blocks:
            for y in range(baseline, -1, -1):
                need_support = True
                block = get_block_type(pix_data[x, y])
                if block is None and final_block < y:
                    data_none[x, y] = 0
                    data_ice[x, y] = 1
                    data_wood[x, y] = 0
                else:
                    data_none[x, y] = 0
                    data_ice[x, y] = 0
                    data_wood[x, y] = 1

    data_total = np.reshape([data_none, data_ice, data_wood], (3, -1))
    np.savetxt(out_path, data_total, delimiter=",", fmt='%1d', encoding='utf8')


if __name__ == "__main__":
    generate("data/generated_data_wgan_gp_5_fz")
