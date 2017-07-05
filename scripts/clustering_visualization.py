"""
author: Fang Ren (SSRL)

7/5/2017
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import os.path

path = '..\\report\\clustering\\'

label1 = loadmat(path + 'particle1_label_20170705.mat')['labels']
label2 = loadmat(path + 'particle2_label_20170705.mat')['labels']
label3 = loadmat(path + 'particle3_label_20170705.mat')['labels']

label1[label1 == 2] = 3.1
label2[label2 == 2] = 3.1
label3[label3 == 2] = 3.1

label1[label1 == 1] = 2.1
label2[label2 == 1] = 2.1
label3[label3 == 1] = 2.1

label1[label1 == 0] = 1.1
label2[label2 == 0] = 1.1
label3[label3 == 0] = 1.1

label1[label1 == 3] = 0.1
label2[label2 == 3] = 0.1
label3[label3 == 3] = 0.1

# visualization
save_path = '..\\report\\clustering\\'
plt.figure(1)
plt.title('particle 1')
plt.imshow(label1, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering1.png'))

plt.figure(2)
plt.title('particle 2')
plt.imshow(label2, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering2.png'))

plt.figure(3)
plt.title('particle 3')
plt.imshow(label3, cmap = 'viridis')
plt.grid('off')
plt.colorbar()
plt.savefig(os.path.join(save_path, 'clustering3.png'))

