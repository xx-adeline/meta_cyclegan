import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random


class TraninDatasets(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        # transform
        transform_list = []
        # resize 286*286
        osize = [opt.load_size, opt.load_size]

        # transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))

        # crop  256*256
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        # 随机水平翻转
        transform_list.append(transforms.RandomHorizontalFlip())
        # 将数据归一化到[0,1]
        transform_list += [transforms.ToTensor()]
        # RGB 3channel的mean、std均为0，5，将数据标准化，更易收敛
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.tran = transforms.Compose(transform_list)

        # 确定train data
        self.datapath = os.path.join(opt.dataroot, opt.datasets)
        self.train_task = os.listdir(self.datapath)
        self.train_task.remove('test')
        self.A_path_list = []
        self.B_path_list = []
        self.label_list = []

        # 将所有图片路径和label载入list
        for i, task in enumerate(self.train_task):
            # task文件路径
            Apath = os.path.join(self.datapath, str(task), 'A')
            Bpath = os.path.join(self.datapath, str(task), 'B')

            # image文件路径
            A_img_name = os.listdir(Apath)
            B_img_name = os.listdir(Bpath)
            self.A_path_list += [os.path.join(Apath, img_name) for img_name in A_img_name]
            self.B_path_list += [os.path.join(Bpath, img_name) for img_name in B_img_name]

        self.A_size = len(self.A_path_list)
        self.B_size = len(self.B_path_list)

    def __getitem__(self, index):
        # 源域和目标域的训练数据数量无需相同
        A_img = self.A_path_list[index % self.A_size]
        # 选择是否固定每次训练数据对
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_img = self.B_path_list[index_B]

        A_img = Image.open(A_img).convert('RGB')
        B_img = Image.open(B_img).convert('RGB')
        A_img = self.tran(A_img)
        B_img = self.tran(B_img)

        return A_img, B_img

    def __len__(self):
        return max(self.A_size, self.B_size)


class TestDatasets(data.Dataset):
    def __init__(self, opt):
        self.opt = opt

        # transform
        transform_list = []
        # resize 286*286
        osize = [opt.load_size, opt.load_size]

        # transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))

        # crop  256*256
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        # 随机水平翻转
        transform_list.append(transforms.RandomHorizontalFlip())
        # 将数据归一化到[0,1]
        transform_list += [transforms.ToTensor()]
        # RGB 3channel的mean、std均为0，5，将数据标准化，更易收敛
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.tran = transforms.Compose(transform_list)

        # 确定train data
        self.datapath = os.path.join(opt.dataroot, opt.datasets, 'test')
        self.test_data = os.listdir(self.datapath)
        self.A_path_list = []
        self.B_path_list = []

        Apath = os.path.join(self.datapath, 'A')
        Bpath = os.path.join(self.datapath, 'B')
        # image文件路径
        A_img_name = os.listdir(Apath)
        B_img_name = os.listdir(Bpath)
        self.A_path_list += [os.path.join(Apath, img_name) for img_name in A_img_name]
        self.B_path_list += [os.path.join(Bpath, img_name) for img_name in B_img_name]
        self.A_size = len(self.A_path_list)
        self.B_size = len(self.B_path_list)
        
    def __getitem__(self, index):
        A_img = self.A_path_list[index]
        B_img = self.B_path_list[index % self.B_size]
        A_img = Image.open(A_img).convert('RGB')
        B_img = Image.open(B_img).convert('RGB')
        A_img = self.tran(A_img)
        B_img = self.tran(B_img)

        return A_img, B_img

    def __len__(self):
        return max(self.A_size, self.B_size)



