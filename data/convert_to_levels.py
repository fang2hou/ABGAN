import os
import time
from sys import stdout, path
import convert.abpype as ap

dirpath, _, filenames = os.walk('../model/results').__next__()

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
    for test_threshold in range(1, 10, 1):
        outpath = '../results/level-{0:02d}-{1:.1f}.xml'.format(index, test_threshold*0.1)
        ap.data_to_level(dirpath+'/'+data, outpath, threshold=test_threshold*0.1)
        num_converted += 1

        progress = num_converted / (num_data * 9) 
        stdout.write("\rProgress: {0:.1f} %".format(100 * progress))
        stdout.flush()

convert_time = time.time() - convert_time

stdout.write("\nDone! Converted {0} data in {1:.3f}s.\n".format(num_data, convert_time))