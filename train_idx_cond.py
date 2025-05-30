import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

import glob
import os

from model.multi_scale_ori_cond import *

class CustomImageIdxDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datas = []
        self.amplitudes = []
        self.shifts = []
        self.idxs = []
        self.class_to_idx = {}

        # 클래스 폴더 순회
        paths = glob.glob(root_dir + '/*.npy')
        for path in paths:
            data = np.load(path)
            amplitude = data.max() - data.min()
            shift = amplitude / 2.
            data = (data - data.min()) / (data.max() - data.min())
            data = 2. * data - 1.

            idx = int(os.path.splitext(os.path.basename(path))[0][-2:])

            self.datas.append(data)
            self.amplitudes.append(amplitude)
            self.idxs.append(idx)
            self.shifts.append(shift)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        data = self.datas[i]
        amplitude = self.amplitudes[i]
        shift = self.shifts[i]
        idx = self.idxs[i]

        return data, amplitude, shift, idx


batch_size = 1024
num_epochs = 3001

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

# Set parameters for this dataset
full_dataset = CustomImageIdxDataset(root_dir='/data/jionkim/signal_dataset/npy_part_res_4000_dir_X_whole')
# 전체 길이와 비율에 따라 분할
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size  # 나머지

# 데이터셋 분할
train_dataset, test_dataset = random_split(
    full_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # 재현 가능성
)

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], condition_size=12)
msresnet = msresnet.cuda()

criterion = nn.L1Loss(size_average=False).cuda()

optimizer = torch.optim.Adam(msresnet.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)
train_loss = np.zeros([num_epochs, 1])
test_loss = np.zeros([num_epochs, 1])
train_acc = np.zeros([num_epochs, 1])
test_acc = np.zeros([num_epochs, 1])

for epoch in range(num_epochs):
    print('Epoch:', epoch)
    msresnet.train()
    scheduler.step()
    # for i, (samples, labels) in enumerate(train_data_loader):
    losses = 0
    for (samples, amps, shifts, idxs) in train_data_loader:
        samplesV = Variable(samples.cuda())
        samplesV = torch.unsqueeze(samplesV, 1)
        amps = amps.squeeze()
        ampsV = Variable(amps.cuda())
        shifts = shifts.squeeze()
        shiftsV = Variable(shifts.cuda())
        idxsV = Variable(idxs.cuda())
        idxsV = torch.unsqueeze(idxsV, 1)

        # Forward + Backward + Optimize
        optimizer.zero_grad()
        predict_data = msresnet(samplesV, idxsV)
        predict_amp = predict_data[:, 0, 0]
        predict_shift = predict_data[:, 0, 1]

        loss_amp = criterion(predict_amp, ampsV)
        loss_shift = criterion(predict_shift, shiftsV)

        loss = loss_amp + loss_shift
        losses += loss.item()

        loss.backward()
        optimizer.step()

    train_loss[epoch] = losses / train_size

    print("Training Loss:", train_loss[epoch])

    msresnet.eval()
    losses_test = 0
    for i, (samples, amps, shifts, idxs) in enumerate(test_data_loader):
        with torch.no_grad():
            samplesV = Variable(samples.cuda())
            samplesV = torch.unsqueeze(samplesV, 1)
            amps = amps.squeeze()
            ampsV = Variable(amps.cuda())
            shifts = shifts.squeeze()
            shiftsV = Variable(shifts.cuda())
            idxsV = Variable(idxs.cuda())
            idxsV = torch.unsqueeze(idxsV, 1)

            predict_data = msresnet(samplesV, idxsV)
            predict_amp = predict_data[:, 0, 0]
            predict_shift = predict_data[:, 0, 1]

            loss_amp = criterion(predict_amp, ampsV)
            loss_shift = criterion(predict_shift, shiftsV)

            loss = loss_amp + loss_shift
            losses_test += loss.item()

    test_loss[epoch] = losses_test / test_size
    print("Test Error:", test_loss[epoch])

    if epoch % 100 == 0:
        torch.save(msresnet, 'weights_cond/weight_epoch_' + str(epoch).zfill(4)  + '.pkl')

plt.plot(train_loss)
plt.show()

plt.plot(test_loss)
plt.show()
