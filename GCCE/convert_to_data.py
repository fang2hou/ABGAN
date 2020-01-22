from level_generator_numpy import generate
import numpy as np
#from PIL import Image

dataset = np.load('snake.npy')
part = dataset[:3000, :]

for count in range(0, 3000):
    pic_data = np.reshape(part[count], (28, 28))

    generate(pic_data, 'data_{0:04d}.txt'.format(count))
