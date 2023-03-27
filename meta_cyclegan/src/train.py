import torch
import os
from model.cyclegan import Cyclegan
from data.dataset import MetaDatset
from options.option import options
from util.read_write_data import makedir
from visualizer import Visualizer


class MetaCyclegan:
    def __init__(self, opt):

        self.opt = opt

        # 创建meta dataset
        self.meta_dataset = MetaDatset(opt)

        # 创建保存模型的路径
        model_root = os.path.join(opt.save_path, 'model')
        makedir(model_root)

        # 创建可视化
        self.visualizer = Visualizer(opt)

        # 创建模型, 或重载模型
        self.model = Cyclegan(opt)
        if self.opt.resume == 'restart':
            self.epoch_start = 1
        else:
            check_point = torch.load(os.path.join(model_root, opt.resume))
            self.model.netG_A.load_state_dict(check_point['generatorA'])
            self.model.netG_B.load_state_dict(check_point['generatorB'])
            self.model.netD_A.load_state_dict(check_point['discriminatorA'])
            self.model.netD_B.load_state_dict(check_point['discriminatorB'])
            self.model.netG_A_meta.load_state_dict(check_point['meta_generatorA'])
            self.model.netG_B_meta.load_state_dict(check_point['meta_generatorB'])
            self.model.netD_A_meta.load_state_dict(check_point['meta_discriminatorA'])
            self.model.netD_B_meta.load_state_dict(check_point['meta_discriminatorB'])
            self.epoch_start = check_point['out_epoch'] + 1

    def train(self):
        for out_epoch in range(self.epoch_start, self.opt.out_epoch_num + 1):
            self.out_epoch = out_epoch
            # 每经过一个外部epoch，衰减一次内部模型的lr
            self.model.update_learning_rate()            
            # 设置内部模型为训练模式，并用外部模型参数初始化内部模型
            self.reset_meta_model()
            # 训练内部模型
            self.meta_training_loop()
            # 间隔固定的out_epoch，tune内部模型后test
            if out_epoch % self.opt.test_interval_epoch == 0:
                self.reset_meta_model()
                self.test_during_train()
                # 将tune后的内部模型也保存下来
                self.checkpoint_model()

    def reset_meta_model(self):
        """设置内部模型为训练模式，并用外部模型参数初始化内部模型"""
        self.model.netG_A_meta.train()
        self.model.netG_A_meta.load_state_dict(self.model.netG_A.state_dict())
        self.model.netG_B_meta.train()
        self.model.netG_B_meta.load_state_dict(self.model.netG_B.state_dict())

        self.model.netD_A_meta.train()
        self.model.netD_A_meta.load_state_dict(self.model.netD_A.state_dict())
        self.model.netD_B_meta.train()
        self.model.netD_B_meta.load_state_dict(self.model.netD_B.state_dict())

    def meta_training_loop(self):
        """随机选择一个task训练内部模型，用内部模型指导外部模型更新"""
        # 抽取一个task用于训练, data是该task中一个batch的数据
        task_dataloader, task_name = self.meta_dataset.sample_train_task()

        # 在log中打印task信息
        self.visualizer.divide_header(task_name, is_train=True)

        # 开始内部训练
        for in_epoch in range(1, self.opt.in_epoch_num + 1):
            for times, [A_img, B_img] in enumerate(task_dataloader):
                A_img = A_img.to(self.opt.device)
                B_img = B_img.to(self.opt.device)
                # 更新内部模型的G,D
                self.model.meta_optimize_parameters(A_img, B_img)

            # 打印loss并保存到log中
            lA, lB = self.model.get_current_losses()
            self.visualizer.print_current_losses(self.out_epoch, in_epoch, lA, lB, is_train=True)
            # 可视化loss
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, lA, 'lossA')
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, lB, 'lossB')
            # 可视化predict
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, self.model.meta_G_pA_ret, 'G_predA')
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, self.model.meta_G_pB_ret, 'G_predB')
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, self.model.meta_D_pA_ret, 'D_predA')
            self.visualizer.plot_line(self.out_epoch, in_epoch / self.opt.in_epoch_num, self.model.meta_D_pB_ret, 'D_predB')

        # 用训练好的内部模型参数, 直接给出外部模型更新的梯度
        for p, meta_p in zip(self.model.netG_A.parameters(), self.model.netG_A_meta.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
            meta_p.to(self.opt.device)
        self.model.optimizer_G.step()

        for p, meta_p in zip(self.model.netG_B.parameters(), self.model.netG_B_meta.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
            meta_p.to(self.opt.device)
        self.model.optimizer_G.step()

        for p, meta_p in zip(self.model.netD_A.parameters(), self.model.netD_A_meta.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
            meta_p.to(self.opt.device)
        self.model.optimizer_D.step()

        for p, meta_p in zip(self.model.netD_B.parameters(), self.model.netD_B_meta.parameters()):
            diff = p - meta_p.cpu()
            p.grad = diff
            meta_p.to(self.opt.device)
        self.model.optimizer_D.step()

    def test_during_train(self):
        """内部模型在tune task微调后, 进行test"""

        # 抽取tune task的全部的数据
        tune_dataloader = self.meta_dataset.sample_tune_task()
        # 在log中打印task信息
        self.visualizer.divide_header('tune', is_train=False)
        # 使用tune data, 开始内部训练
        for in_epoch in range(1, self.opt.tune_epoch_num + 1):
            for times, [A_img, B_img] in enumerate(tune_dataloader):
                A_img = A_img.to(self.opt.device)
                B_img = B_img.to(self.opt.device)
                # 更新内部模型的G,D
                self.model.meta_optimize_parameters(A_img, B_img)
            # 打印loss并保存到log中
            lA, lB = self.model.get_current_losses()
            self.visualizer.print_current_losses(self.out_epoch, in_epoch, lA, lB, is_train=False)

        # 抽取test数据
        test_dataset, _ = self.meta_dataset.sample_test()
        # 使用test data, 测试内部模型
        test_A_img, test_B_img = test_dataset[opt.test_index]
        test_A_img = test_A_img.to(self.opt.device).unsqueeze(0)
        test_B_img = test_B_img.to(self.opt.device).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            self.model.meta_forward(test_A_img, test_B_img)
        self.model.train()
        # 保存测试的结果图片
        self.visualizer.save_images(self.model.get_current_visuals(), self.out_epoch)

    def checkpoint_model(self):
        """保存当前内、外部模型参数"""
        # 保存外部模型参数
        checkpoint = {'generatorA': self.model.netG_A.state_dict(),
                      # 'generatorB': self.model.netG_B.state_dict(),
                      # 'discriminatorA': self.model.netD_A.state_dict(),
                      # 'discriminatorB': self.model.netD_B.state_dict(),
                      'out_epoch': self.out_epoch
                      }
        # 保存内部模型参数
        checkpoint = {'meta_generatorA': self.model.netG_A_meta.state_dict(),
                      # 'meta_generatorB': self.model.netG_B_meta.state_dict(),
                      # 'discriminatorA': self.model.netD_A_meta.state_dict(),
                      # 'discriminatorB': self.model.netD_B_meta.state_dict(),
                      'out_epoch': self.out_epoch
                      }
        filename = os.path.join(self.opt.save_path, 'model/epoch_{}.pth.tar'.format(self.out_epoch))
        torch.save(checkpoint, filename)


if __name__ == '__main__':
    opt = options().opt
    MC = MetaCyclegan(opt)
    MC.train()

