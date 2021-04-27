import os
import torch

from PIL.Image import open
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from models import utils


class ImageImageLoader(Dataset):

    def __init__(self,
                 transform_list,
                 train_x_path,
                 train_y_path):

        self.transform_list = transform_list

        x_img_name = os.listdir(train_x_path)
        y_img_name = os.listdir(train_y_path)

        self.x_img_path = []
        self.y_img_path = []

        x_img_name = sorted(x_img_name)
        y_img_name = sorted(y_img_name)

        img_paths = zip(x_img_name, y_img_name)
        for item in img_paths:
            self.x_img_path.append(train_x_path + os.sep + item[0])
            self.y_img_path.append(train_y_path + os.sep + item[1])

        assert len(self.x_img_path) == len(self.y_img_path), 'Images in directory must have same file indices!!'

        self.len = len(x_img_name)

        del x_img_name
        del y_img_name

    def __getitem__(self, index):
        new_img_x = open(self.x_img_path[index]).convert('RGB')
        new_img_y = open(self.y_img_path[index]).convert('RGB')

        # print(self.x_img_path[index])     # assert wx == wy and hx == hy, 'image size should be same'

        new_img_x = utils.m_invert(new_img_x)
        new_img_y = utils.m_invert(new_img_y)

        transform_list_temp = self.transform_list

        width, height = new_img_x.size
        if width <= 512 or height <= 512:   # the number of width and height must match to below composer
            for item in transform_list_temp:
                if type(item) is transforms.RandomCrop:
                    transform_list_temp.remove(item)

        transform = transforms.Compose(transform_list_temp)

        out_img_x = transform(new_img_x)
        out_img_y = transform(new_img_y)

        new_img_x_deu = utils.Datonize.deuteranopia_img(new_img_x)
        out_img_x_deu = transform(new_img_x_deu)

        new_img_x_diff = utils.make_diff(new_img_x, new_img_x_deu)
        out_img_x_diff = transform(new_img_x_diff)

        out_img_x_feat = torch.cat((out_img_x_deu, out_img_x_diff))

        new_img_y_deu = utils.Datonize.deuteranopia_img(new_img_y)
        out_img_y_deu = transform(new_img_y_deu)

        new_img_y_diff = utils.make_diff(new_img_y, new_img_y_deu)
        out_img_y_diff = transform(new_img_y_diff)

        out_img_y_feat = torch.cat((out_img_y_deu, out_img_y_diff))

        return (out_img_x, out_img_x_feat), (out_img_y, out_img_y_feat)

    def __len__(self):
        return self.len


class CUD_TrainLoader:

    def __init__(self,
                 dataset_path,
                 label_path,
                 batch_size=64,
                 num_workers=0,
                 pin_memory=True):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_data_path = dataset_path
        self.train_label_path = label_path

        transform_list = list()
        transform_list.append(transforms.RandomCrop((512, 512)))
        transform_list.append(transforms.ToTensor())

        # use your own data loader
        self.Loader = DataLoader(ImageImageLoader(transform_list,
                                                  self.train_data_path,
                                                  self.train_label_path),
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 shuffle=True,
                                 pin_memory=self.pin_memory)

    def __len__(self):
        return self.Loader.__len__()
