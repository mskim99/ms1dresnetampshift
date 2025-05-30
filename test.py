import numpy as np
from torch.autograd import Variable
import os
from torch.utils.data import Dataset, DataLoader, random_split
import glob

from model.multi_scale_ori import *

class CustomImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.datas = []
        self.amplitudes = []
        self.shifts = []
        self.class_to_idx = {}

        # 클래스 폴더 순회
        paths = glob.glob(root_dir + '/*.npy')
        for path in paths:
            data = np.load(path)
            amplitude = data.max() - data.min()
            shift = amplitude / 2.

            self.datas.append(data)
            self.amplitudes.append(amplitude)
            self.shifts.append(shift)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        amplitude = self.amplitudes[idx]
        shift = self.shifts[idx]

        return data, amplitude, shift

os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'

# Set parameters for this dataset
full_dataset = CustomImageDataset(root_dir='/data/jionkim/signal_dataset/npy_part_res_4000_dir_X')
# 전체 길이와 비율에 따라 분할
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
test_size = total_size - train_size  # 나머지

# 데이터셋 분할
_, test_dataset = random_split(
    full_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(42)  # 재현 가능성
)

# train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

# msresnet = MSResNet(input_channel=1, layers=[1, 1, 1, 1], num_classes=6)
msresnet = torch.load('weights_load/weight_epoch_2900.pkl', weights_only=False)
msresnet = msresnet.cuda()
msresnet.eval()

criterion = nn.L1Loss(size_average=False).cuda()

losses_test = 0
sample_error = 0

with open("predicted_unnorm.txt", "w", encoding="utf-8") as file:
    file.write('pred (amp) / real (amp) / pred (shift) / real (shift)\n')

for i, (samples, amps, shifts) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        samplesV = (samplesV - samplesV.min()) / (samplesV.max() - samplesV.min())
        samplesV = 2. * samplesV - 1.
        samplesV = torch.unsqueeze(samplesV, 1)
        amps = amps.squeeze()
        ampsV = Variable(amps.cuda())
        shifts = shifts.squeeze()
        shiftsV = Variable(shifts.cuda())

        predict_data = msresnet(samplesV)
        predict_amp = predict_data[:, 0]
        predict_shift = predict_data[:, 1]

        loss_amp = criterion(predict_amp, ampsV)
        loss_shift = criterion(predict_shift, shiftsV)

        loss = loss_amp + loss_shift
        losses_test += loss.item()

    # Print amplitude as txt
    with open("predicted_unnorm.txt", "a", encoding="utf-8") as file:
        file.write(str(predict_amp.item()) + ' / ' + str(amps.item()) + ' / ' + str(predict_shift.item()) + ' / ' + str(shifts.item()) + '\n')

    # print amplitude x samples
    samples_output = predict_amp.item() * samples + predict_shift.item()
    sample_error += (samples_output - samples).mean()
    np.save('pred_signal/signal_' + str(i).zfill(3) + '_real.npy', samples)
    np.save('pred_signal/signal_' + str(i).zfill(3) + '_pred.npy', samples_output)

print("Total Amplitude Error:", (losses_test / test_size))
print("Total Sample Error:", (sample_error / test_size).item())

with open("predicted_unnorm.txt", "a", encoding="utf-8") as file:
    file.write('\n')
    file.write("Total Amplitude Error:" + str(losses_test / test_size))
    file.write("Total Sample Error:" + str(sample_error / test_size).item())
