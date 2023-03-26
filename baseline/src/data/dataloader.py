from data.dataset import TraninDatasets, TestDatasets
from torch.utils import data


def get_train_dataloader(opt):
    dataset = TraninDatasets(opt)
    dataloader = data.DataLoader(
        dataset,
        batch_size=opt.batchsize,
        num_workers=int(opt.num_threads),
        shuffle=True,
    )
    print('The number of training images = %d' % len(dataset))
    return dataloader, len(dataset)


def get_test_dataloader(opt):
    dataset = TestDatasets(opt)
    dataloader = data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=int(opt.num_threads),
        shuffle=False,
    )
    print('The number of testing images = %d' % len(dataset))
    return dataloader, len(dataset)
