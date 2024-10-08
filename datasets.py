from torchvision import transforms
import os, random, argparse, sys
from torch.utils.data import Dataset, DataLoader

# 将上级目录加入 sys.path， 防止命令行运行时找不到包
curPath = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(curPath)[0]
sys.path.append(root_path)

from PIL import Image
import numpy as np

from MY_DICT import DICT


class pedCls_Dataset(Dataset):
    '''
        读取多个数据集的数据
    '''

    def __init__(self, dict, ds_name_list, txt_name, get_num):
        self.dict = dict
        self.base_dir_list = [self.dict[ds_name] for ds_name in ds_name_list]
        self.txt_name = txt_name
        self.image_transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.get_num = get_num
        self.images, self.labels = self.initImgLabel()

    def initImgLabel(self):
        '''
            读取图片 和 label
        '''
        images = []
        labels = []

        for base_dir in self.base_dir_list:
            txt_path = os.path.join(base_dir, 'dataset_txt', self.txt_name)
            with open(txt_path, 'r') as f:
                data = f.readlines()

            for line in data:
                line = line.replace('\\', os.sep)
                line = line.strip().split()
                image_path = os.path.join(base_dir, line[0])
                label = line[-1]
                images.append(image_path)
                labels.append(label)

        if self.get_num == -1:
            images = images
            labels = labels
        else:
            images = images[: self.get_num]
            labels = labels[: self.get_num]

        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        label = self.labels[idx]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        img = self.image_transformer(img)
        return img, label


if __name__ == '__main__':
    ped_ds = pedCls_Dataset(dict=DICT, ds_name_list=['D1'], txt_name='augmentation_train.txt', get_num=-1)
    ped_loader = DataLoader(ped_ds, batch_size=4, shuffle=True, drop_last=True)
    print(len(ped_ds))

















