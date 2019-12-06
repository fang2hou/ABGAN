import torch
import sys
import models.dcgan as dcgan
import numpy as np

# Check the parameter is legal or not
if 1 == len(sys.argv) or ".pth" != sys.argv[1][-4:]:
    saved_model = "saves/netG_epoch_750_32.pth"
    #exit('USE: decoder.py MODEL.pth')
else:
    saved_model = sys.argv[1]

# Set number of levels
batch_size = 10

# Set the parameters same as the used in training process
map_size = 128
nz = 32
ngf = 32
ngpu = 2
n_extra_layers = 0
z_dims = 21

threshold = 0.7

# Load Generator
netG = dcgan.DCGAN_G(map_size, nz, z_dims, ngf, ngpu, n_extra_layers)
netG.load_state_dict(torch.load(saved_model))
noise = torch.Tensor(batch_size, nz, 1, 1).normal_(0, 1)

# GPU Acceleration
if torch.cuda.is_available():
    noise = noise.cuda()
    netG = netG.cuda()

# Generate
results = netG(noise).data

for i, result in enumerate(results, 1):
    temp = np.zeros((z_dims, map_size * map_size), dtype=float)

    for x in range(map_size):
        for y in range(map_size):
            temp_binary_type = ''
            reliablity_one = []
            reliablity_zero = []

            for i in result[:, x, y]:
                if i > threshold:
                    temp_binary_type += '1'
                    reliablity_one.append(i)
                else:
                    temp_binary_type += '0'
                    reliablity_zero.append(i)

            channel = int(temp_binary_type, 2)
            channel_value = sum(reliablity_one)/len(reliablity_one) - \
                sum(reliablity_zero)/len(reliablity_zero)
            index = x * map_size + y
            temp[channel][index] = channel_value

    np.savetxt('data/generated_data_wgan_5/from_net_{0:02d}.gz'.format(i),
               temp, delimiter=",", fmt='%.2f', encoding='utf8')
