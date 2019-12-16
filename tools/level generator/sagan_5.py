import torch
import sys
import abganlibs.models.sagan as dcgan
import numpy as np

# Check the parameter is legal or not
if 1 == len(sys.argv) or ".pth" != sys.argv[1][-4:]:
    saved_model = "saves/SAGAN_5/netG_epoch_700_32.pth"
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
z_dims = 5

threshold = .5

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
    temp = np.zeros((21, map_size * map_size), dtype=float)

    for x in range(map_size):
        for y in range(map_size):
            temp_binary_type = ''
            reliablity_one = []
            reliablity_zero = []

            temp_max = result[:, x, y].max()
            temp_result = result[:, x, y] / temp_max

            for j in temp_result:
                if j > threshold:
                    temp_binary_type += '1'
                    reliablity_one.append(j)
                else:
                    temp_binary_type += '0'
                    reliablity_zero.append(j)

            channel = int(temp_binary_type, 2)
            if channel < 21 and channel >= 0:
                if len(reliablity_one) != 0:
                    one_avg = sum(reliablity_one) / len(reliablity_one)
                else:
                    one_avg = 0

                if len(reliablity_zero) != 0:
                    zero_avg = sum(reliablity_zero) / len(reliablity_zero)
                else:
                    zero_avg = 0

                channel_value = one_avg - zero_avg
                index = x * map_size + y
                temp[channel][index] = channel_value * temp_max

    np.savetxt('data/generated_data_sagan_5/from_net_{0:02d}.gz'.format(i),
               temp, delimiter=",", fmt='%.2f', encoding='utf8')
