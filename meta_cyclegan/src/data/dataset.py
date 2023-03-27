import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import random


class MetaDatset:
    def __init__(self, opt):
        self.opt = opt

        # 确定train task, 默认‘tunetask’文件名为tune task
        self.datapath = os.path.join(opt.dataroot, opt.datasets)
        self.train_task = os.listdir(self.datapath)
        self.train_task.remove('test')

    def sample_train_task(self):
        """随机抽取一个task, 以其抽样数据构建dataset和dataloader"""
        # 随机抽取一个task
        task = random.sample(self.train_task, 1)[0]
        Apath = os.path.join(self.datapath, task, 'A')
        Bpath = os.path.join(self.datapath, task, 'B')

        # 从该task路径下随机抽样task size数据作为本task训练数据
        A_img_name = os.listdir(Apath)
        A_img_name = random.sample(A_img_name, min(self.opt.tasksize, len(A_img_name)))
        A_img_path = [os.path.join(self.datapath, task, 'A', img_name) for img_name in A_img_name]

        B_img_name = os.listdir(Bpath)
        B_img_name = random.sample(B_img_name, min(self.opt.tasksize, len(B_img_name)))
        B_img_path = [os.path.join(self.datapath, task, 'B', img_name) for img_name in B_img_name]

        # 以本次抽样数据构建task dataset和dataloader
        task_dataset = TaskDataset(self.opt, A_img_path, B_img_path, is_train=True)
        dataloader = torch.utils.data.DataLoader(
            task_dataset,
            batch_size=self.opt.batchsize,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        print('sample: %s, The number of training data for the current task = %d' % (task, len(task_dataset)))
        return dataloader, task

    def sample_tune_task(self):
        """抽样tune task的全部数据, 构建dataset和dataloader"""
        task = 'tunetask'
        Apath = os.path.join(self.datapath, task, 'A')
        Bpath = os.path.join(self.datapath, task, 'B')
        A_img_name = os.listdir(Apath)
        A_img_path = [os.path.join(self.datapath, task, 'A', img_name) for img_name in A_img_name]

        B_img_name = os.listdir(Bpath)
        B_img_path = [os.path.join(self.datapath, task, 'B', img_name) for img_name in B_img_name]

        tune_dataset = TaskDataset(self.opt, A_img_path, B_img_path, is_train=True)
        dataloader = torch.utils.data.DataLoader(
            tune_dataset,
            batch_size=self.opt.batchsize,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        print('sample: %s, The number of training data for the tune task = %d' % (task, len(tune_dataset)))
        return dataloader

    def sample_test(self):
        """抽样test的全部数据, 构建dataset和dataloader"""
        task = 'test'
        Apath = os.path.join(self.datapath, task, 'A')
        Bpath = os.path.join(self.datapath, task, 'B')
        A_img_name = os.listdir(Apath)
        A_img_path = [os.path.join(self.datapath, task, 'A', img_name) for img_name in A_img_name]

        B_img_name = os.listdir(Bpath)
        B_img_path = [os.path.join(self.datapath, task, 'B', img_name) for img_name in B_img_name]

        test_dataset = TaskDataset(self.opt, A_img_path, B_img_path, is_train=False)
        dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=int(self.opt.num_threads))
        print('sample: %s, The number of training images = %d' % (task, len(test_dataset)))
        return test_dataset, dataloader


class TaskDataset(data.Dataset):
    def __init__(self, opt, A_img_path, B_img_path, is_train=True):
        """以某一个task的抽样数据构建的dataset"""
        self.opt = opt

        # transform
        transform_list = []
        osize = [opt.load_size, opt.load_size]
        # transform_list.append(transforms.Resize(osize, transforms.InterpolationMode.BICUBIC))
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        # crop  256*256
        transform_list.append(transforms.RandomCrop(opt.crop_size))
        # test时不flip
        if is_train:
            transform_list.append(transforms.RandomHorizontalFlip())
        # norm
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.tran = transforms.Compose(transform_list)

        self.A_paths = A_img_path
        self.B_paths = B_img_path

        # 源域和目标域task dataset大小
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        # 源域和目标域的训练数据数量无需相同
        A_path = self.A_paths[index % self.A_size]
        # 选择是否固定每次训练数据对, 默认不固定
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.tran(A_img)
        B = self.tran(B_img)

        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)



