from options.option import options
from data.dataset import MetaDatset
from model.cyclegan import Cyclegan
import torch
import os
from util.read_write_data import makedir
from util import util


def test(opt):
    # 创建test dataset
    meta_dataset = MetaDatset(opt)
    _, test_dataloader = meta_dataset.sample_test()

    # 创建测试结果的路径
    test_root_B = os.path.join(opt.save_path, 'test', 'B')
    makedir(test_root_B)

    # 重载模型
    model = Cyclegan(opt)
    model_root = os.path.join(opt.save_path, 'model')
    check_point = torch.load(os.path.join(model_root, opt.resume))
    model.netG_A_meta.load_state_dict(check_point['generatorA'])

    for times, [A_img, _] in enumerate(test_dataloader):
        model.eval()
        A_img = A_img.to(opt.device)
        with torch.no_grad():
            test_fake_B = model.netG_A_meta(A_img)
        numpy_fake_B = util.tensor2im(test_fake_B)
        img_path_B = os.path.join(test_root_B, 'index%.3d_%s.png' % (times, "B"))
        util.save_image(numpy_fake_B, img_path_B)


if __name__ == '__main__':
    opt = options().opt
    test(opt)


