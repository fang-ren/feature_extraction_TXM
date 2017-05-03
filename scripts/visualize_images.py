"""
author: Fang Ren (SSRL)

5/1/2017
"""
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os.path

raw_path = '..\\data\\Imgstack.mat'
nor_path = '..\\data\\Imgstack_Nor.mat'
energy_path = '..\\data\\Imgstack_energy.mat'

images = loadmat(raw_path)['Imgstack']
images_norm = loadmat(nor_path)['Nor']
energies = loadmat(energy_path)['Es']


# print images.shape, energies.shape

# visualize raw images
save_path = '..\\report\\'
plt.figure(1)
plt.subplot(221)
plt.title('1st raw images')
plt.imshow(images[:,:,0])
#plt.clim(0, 1)
plt.colorbar()
plt.subplot(222)
plt.title('26th raw images')
plt.imshow(images[:,:,25])
#plt.clim(0, 1)
plt.colorbar()
plt.subplot(223)
plt.title('51st raw images')
plt.imshow(images[:,:,50])
#plt.clim(0, 1)
plt.colorbar()
plt.subplot(224)
plt.title('76th raw images')
plt.imshow(images[:,:,75])
#plt.clim(0, 1)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'raw_images.png'))


# visualize normalized images

plt.figure(2)
plt.subplot(221)
plt.title('1st normalized images')
plt.imshow(images_norm[:,:,0])
plt.clim(0, 1)
plt.colorbar()
plt.subplot(222)
plt.title('26th normalized images')
plt.imshow(images_norm[:,:,25])
plt.clim(0, 1)
plt.colorbar()
plt.subplot(223)
plt.title('51st normalized images')
plt.imshow(images_norm[:,:,50])
plt.clim(0, 1)
plt.colorbar()
plt.subplot(224)
plt.title('76th normalized images')
plt.imshow(images_norm[:,:,75])
plt.clim(0, 1)
plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'normalized_images.png'))
plt.close('all')