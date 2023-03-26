import argparse
import torch
import os


class options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        # base
        parser.add_argument('--save_path', default='../checkpoint', )
        parser.add_argument('--name', default='cyclegan')
        parser.add_argument('--GPU_id', type=str, default='0', help='choose GPU ID [0 1]')
        parser.add_argument('--resume', default='restart')
        # datasets
        parser.add_argument('--dataroot', default='../../datasets')
        parser.add_argument('--datasets', default='juanben_task')
        parser.add_argument('--num_threads', default=4, type=int, help='threads for loading data')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--batchsize', default=1)
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        # train
        parser.add_argument('--lr', type=float, default=0.0002, help='outer learning rate for adam')
        parser.add_argument('--epoch_num', default=200)
        parser.add_argument('--n_epochs_keep', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        # test
        parser.add_argument('--test_index', type=int, default=0, help="test which data")
        # model
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--pool_size', type=int, default=50,
                            help='the size of image buffer that stores previously generated images')

        self.opt = parser.parse_args()
        self.opt.device = torch.device('cuda:{}'.format(self.opt.GPU_id))
        self.opt.save_path = os.path.join(self.opt.save_path, self.opt.name)


