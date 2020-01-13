import os
import time
from sys import stdout, path
import abganlibs.abpype as ap

data_directory = 'data/generated_data_DAnet_21/'
level_directory = 'data/generated_levels_DAnet_21/'

#data_directory = 'data/generated_data_wgan_gp_5/'
#level_directory = 'data/generated_levels_wgan_gp_5/'

dirpath, _, filenames = os.walk(data_directory).__next__()

data_list = []

# Filter
for filename in filenames:
    if filename[-3:] == '.gz':
        data_list.append(filename)

num_data = len(data_list)
print('{} generated levels found.'.format(num_data))

convert_time = time.time()

num_converted = 0
for index, data in enumerate(data_list, 3):
    outpath = level_directory + \
        'level-{0:02d}.xml'.format(index)
    ap.data_to_level_ruby(dirpath + '/' + data, outpath)
    num_converted += 1

    progress = num_converted / num_data
    stdout.write("\rProgress: {0:.1f} %".format(100 * progress))
    stdout.flush()

convert_time = time.time() - convert_time

stdout.write("\nDone! Converted {0} data in {1:.3f}s.\n".format(
    num_data, convert_time))
