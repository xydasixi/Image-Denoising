import cv2
import numpy as np
import glob
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch
from skimage import io, img_as_float, img_as_ubyte
import numpy as np


patch_size, stride = 40, 10
# patch_size, stride = 160, 20
aug_times = 1
scales = [1, 0.9, 0.8, 0.7]
batch_size = 128


class GetDataset(Dataset):
    def __init__(self, xs, sigma):
        super(GetDataset, self).__init__()
        self.xs = xs
        self.sigma = sigma

    def __getitem__(self, index):
        batch_x = self.xs[index]
        noise = torch.randn(batch_x.size()).mul_(self.sigma / 255.0)
        batch_y = batch_x + noise
        return batch_y, batch_x

    def __len__(self):
        return self.xs.size(0) # size(0) 返回当前张量维数的第一维


def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name):
    # get multiscale patches from a single image
    img = cv2.imread(file_name, 0)  # gray scale
    h, w = img.shape
    patches = []
    for s in scales:
        h_scaled, w_scaled = int(h*s), int(w*s)
        img_scaled = cv2.resize(img, (h_scaled, w_scaled), interpolation=cv2.INTER_CUBIC)
        # extract patches
        for i in range(0, h_scaled-patch_size+1, stride):
            for j in range(0, w_scaled-patch_size+1, stride):
                x = img_scaled[i:i+patch_size, j:j+patch_size]
                for k in range(0, aug_times):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)
    return patches

def datagenerator(data_path, sigma = 15):
    # generate clean patches from a dataset
    file_list = glob.glob(data_path + '/*.png')  # get name list of all .png files
    # initrialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i])
        for patch in patches:
            data.append(patch)
    data = np.array(data, dtype='uint8')
    data = np.expand_dims(data, axis=1)
    discard_n = len(data) - len(data) // batch_size * batch_size  # because of batch namalization
    data = np.delete(data, range(discard_n), axis=0)
    return data

def Gaussian_noise(sigma,image):
    mean = 0
    sigma = sigma/255
    image = img_as_float(image)
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 1.0)

    noisy = img_as_ubyte(noisy)
    return noisy

