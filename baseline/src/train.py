from options.option import options
from data.dataloader import get_train_dataloader
from data.dataset import TestDatasets
from model.cyclegan import Cyclegan
import torch
import os
from util.read_write_data import makedir
from visualizer import Visualizer


def train(opt):
    # 创建dataloader
    train_dataloader, train_data_len = get_train_dataloader(opt)
    test_dataset = TestDatasets(opt)

    # 创建保存模型的路径
    model_root = os.path.join(opt.save_path, 'model')
    makedir(model_root)

    # 创建可视化
    visualizer = Visualizer(opt)

    # 创建模型，或重载模型
    model = Cyclegan(opt)
    if opt.resume == 'restart':
        epoch_start = 1
    else:
        check_point = torch.load(os.path.join(model_root, opt.resume))
        model.netG_A.load_state_dict(check_point['generatorA'])
        model.netG_B.load_state_dict(check_point['generatorB'])
        model.netD_A.load_state_dict(check_point['discriminatorA'])
        model.netD_B.load_state_dict(check_point['discriminatorB'])
        epoch_start = check_point['epoch'] + 1

    # 开始训练
    for epoch in range(epoch_start, opt.n_epochs_keep + opt.n_epochs_decay + 1):
        model.train()
        # 学习率衰减
        model.update_learning_rate()

        for times, [A_img, B_img] in enumerate(train_dataloader):
            A_img = A_img.to(opt.device)
            B_img = B_img.to(opt.device)
            # 更新G,D
            model.optimize_parameters(A_img, B_img)
            if times % 200 == 0:
                lA, lB = model.get_current_losses()
                visualizer.print_current_losses(epoch, times, lA, lB)
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, lA, "lossA")
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, lB, "lossB")
                # 可视化predict
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, model.G_pA_ret, 'G_predA')
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, model.G_pB_ret, 'G_predB')
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, model.D_pA_ret, 'D_predA')
                visualizer.plot_line(epoch, (float(times) * opt.batchsize) / train_data_len, model.D_pB_ret, 'D_predB')

        test_A_img, test_B_img = test_dataset[opt.test_index]
        test_A_img = test_A_img.to(opt.device).unsqueeze(0)
        test_B_img = test_B_img.to(opt.device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            model.forward(test_A_img, test_B_img)
        model.train()
        visualizer.save_images(model.get_current_visuals(), epoch)

        # 保存模型参数
        if epoch > 150:
            state = {'generatorA': model.netG_A.state_dict(),
                     # 'generatorB': model.netG_B.state_dict(),
                     # 'discriminatorA': model.netD_A.state_dict(),
                     # 'discriminatorB': model.netD_B.state_dict(),
                     'epoch': epoch
                     }

            filename = os.path.join(opt.save_path, 'model/epoch_{}.pth.tar'.format(epoch))
            torch.save(state, filename)


if __name__ == '__main__':
    opt = options().opt
    train(opt)
