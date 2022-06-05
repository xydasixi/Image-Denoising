import argparse
import os, time, datetime, glob
# import PIL.Image as Image
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread, imsave


LR = 0.001
EPOCH = 5
SIGMA = 25
BATCH_SIZE = 128
DATA_PATH = 'data/Test/Set68'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join('models', 'DnCNN'+'_' + 'sigma' + str(SIGMA))



class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__ == '__main__':
    file_list = glob.glob(MODEL_PATH + '/*.pth')
    model = torch.load(file_list[-1])
    model.eval()
    model = model.to(DEVICE)
    psnrs = []
    ssims = []

    for im in os.listdir(DATA_PATH):
        if im.endswith(".jpg") or im.endswith(".bmp") or im.endswith(".png"):
            x = np.array(imread(os.path.join(DATA_PATH, im)), dtype=np.float32)/255.0
            y = x + np.random.normal(0, SIGMA / 255.0, x.shape)  # Add Gaussian noise without clipping
            y = y.astype(np.float32)
            y_ = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
            start_time = time.time()
            y_ = y_.to(DEVICE)
            x_ = model(y_)
            x_ = x_.view(y.shape[0], y.shape[1])
            psnr_temp = peak_signal_noise_ratio(x, x_)
            ssim_temp = structural_similarity(x, x_)
            psnrs.append(psnr_temp)
            ssims.append(ssim_temp)
    psnr_avg = np.mean(psnrs)
    ssim_avg = np.mean(ssims)
    psnrs.append(psnr_avg)
    ssims.append(ssim_avg)
    np.savetxt('results/test_result.txt' , np.hstack((psnrs, ssims)),fmt='%2.4f')

