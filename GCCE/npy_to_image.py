from level_generator_numpy import generate
import numpy as np
#import scipy.misc
import imageio
from PIL import Image

dataset = np.load('snake.npy')
part = dataset[:1, :]
print(part.shape)

for count in range(0, part.shape[0]):
    pic_data = np.reshape(part[count], (28, 28))
    print(pic_data.shape)
    #B = part[count, :, :]
    img = Image.fromarray(pic_data)
    imageio.imsave("image/img" + str(count) + ".png", img)

    # generate(pic_data, 'data_{0:04d}.txt'.format(count))
