import numpy as np
import os, glob, datetime, time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import data_processing as dp

DEPTH = 14
LR = 0.001
EPOCH = 10
SIGMA = 25
BATCH_SIZE = 128
DATA_PATH = 'data/Train400'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_dir = os.path.join('models', 'DnCNN'+'_' + 'sigma' + str(SIGMA))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

class DnCNN(torch.nn.Module):
    def __init__(self, depth=DEPTH, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers += [nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding),
                   nn.ReLU(inplace=True)]
        for i in range(depth-2):
            layers += [nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,padding=padding, bias=False),
                       nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                       nn.ReLU(inplace=True)]#卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        layers += [nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)]
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    def forward(self, x):
        y = x
        out = self.network(x)
        return y - out
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = DnCNN()
    model.train()
    criterion = nn.MSELoss(reduction = 'sum')
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    for epoch in range(0, EPOCH):
        scheduler.step(epoch)  # step to the learning rate in this epcoh
        data = dp.datagenerator(data_path=DATA_PATH, sigma = SIGMA)
        data = data.astype('float32')/255.0
        data = torch.from_numpy(data)  # tensor of the clean patches, NXCXHXW
        DDataset = dp.GetDataset(data, SIGMA)
        DLoader = DataLoader(dataset=DDataset, num_workers=4, drop_last=True, batch_size=BATCH_SIZE, shuffle=True)
        epoch_loss = 0
        start_time = time.time()
        for n_count, (y, x) in enumerate(DLoader):
            batch_x, batch_y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_y), batch_x)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, data.size(0)//BATCH_SIZE, loss.item()/BATCH_SIZE))
        elapsed_time = time.time() - start_time

        np.savetxt('results/train_results/depth_%dtrain_result%03d.txt'% (DEPTH,epoch+1), np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
        # torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
        torch.save(model, os.path.join(save_dir, 'depth_%dmodel_%03d.pth' % (DEPTH,epoch+1)))
