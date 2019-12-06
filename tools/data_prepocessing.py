import os
import time
from sys import stdout
from tools import abpype as ap

data_directory = 'data/original_data_21/'
level_directory = 'data/original_levels/'

dirpath, _, filenames = os.walk(level_directory).__next__()

level_list = []

for filename in filenames:
    if filename[-4:] == '.xml':
        level_list.append(filename)

num_levels = len(level_list)
print('{} Science Birds levels found.'.format(num_levels))

convert_time = time.time()

num_converted = 0
for level in level_list:
    ap.level_to_data(dirpath + '/' + level,
                     data_directory + level[:-4] + '.gz')
    num_converted += 1

    progress = num_converted / num_levels * 100
    stdout.write("\rProgress: %.1f %%" % (progress))
    stdout.flush()

convert_time = time.time() - convert_time

stdout.write("\nDone! Converted {0} Science Birds levels in {1:.3f}s.".format(
    num_levels, convert_time))
