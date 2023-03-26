from options.option import options
from data.dataloader import get_test_dataloader
from model.cyclegan import Cyclegan
import torch
import os
from util.read_write_data import makedir
from util import util


def test(opt):
    # 创建dataloader
    test_dataloader, _ = get_test_dataloader(opt)

    # 创建测试结果的路径
    test_root_A = os.path.join(opt.save_path, 'test', 'A')
    test_root_B = os.path.join(opt.save_path, 'test', 'B')
    makedir(test_root_A)
    makedir(test_root_B)

    # 重载模型
    model = Cyclegan(opt)
    model_root = os.path.join(opt.save_path, 'model')
    check_point = torch.load(os.path.join(model_root, opt.resume))
    model.netG_A.load_state_dict(check_point['generatorA'])
    model.netG_B.load_state_dict(check_point['generatorB'])

    for times, [A_img, B_img] in enumerate(test_dataloader):
        model.train()
        A_img = A_img.to(opt.device)
        B_img = B_img.to(opt.device)
        with torch.no_grad():
            model.forward(A_img, B_img)
        image_numpy_A = util.tensor2im(A_img)
        image_numpy_B = util.tensor2im(B_img)
        img_path_A = os.path.join(test_root_A, 'index%.3d_%s.png' % (times, "A"))
        img_path_B = os.path.join(test_root_B, 'index%.3d_%s.png' % (times, "B"))
        util.save_image(image_numpy_A, img_path_A)
        util.save_image(image_numpy_B, img_path_B)


if __name__ == '__main__':
    opt = options().opt
    test(opt)


