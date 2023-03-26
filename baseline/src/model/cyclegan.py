from torch import nn
import torch
from . import networks
import itertools
from model.image_pool import ImagePool
from collections import OrderedDict


class Cyclegan(nn.Module):
    def __init__(self, opt):
        super(Cyclegan, self).__init__()
        self.opt = opt
        # 创建源域/目标域 生成器
        self.netG_A = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain).to(opt.device)
        self.netG_B = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain).to(opt.device)
        # 创建源域/目标域 判别器
        self.netD_A = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain).to(opt.device)
        self.netD_B = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain).to(opt.device)

        # 判别器以50%概率使用历史生成图像，以50%概率使用当前生成图像
        self.fake_A_pool = ImagePool(opt.pool_size)
        self.fake_B_pool = ImagePool(opt.pool_size)

        # 创建损失函数
        self.criterionGAN = networks.GANLoss().to(opt.device)
        self.criterionCycle = torch.nn.L1Loss().to(opt.device)

        # 创建优化器
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers = []
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        # 给每个优化器, 创建学习率衰减器
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def forward(self, img_A, img_B):
        self.real_A = img_A
        self.real_B = img_B

        self.fake_B = self.netG_A(self.real_A)      # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)       # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)      # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)       # G_A(G_B(B))

    def optimize_parameters(self, img_A, img_B):
        # forward
        self.forward(img_A, img_B)
        # 关闭D的梯度
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        # 计算G的loss
        self.backward_G()
        self.optimizer_G.step()
        # 打开D的梯度      fake_image.detach()不会回传梯度到G
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        # 计算D的loss
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        # Real
        self.pred_A_real = self.netD_A(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_A_real, True)
        # Fake
        self.pred_A_fake = self.netD_A(fake_B.detach())     # fake_image.detach()不会回传梯度到G
        loss_D_fake = self.criterionGAN(self.pred_A_fake, False)

        self.loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_A.backward()

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        # Real
        self.pred_B_real = self.netD_B(self.real_A)
        loss_D_real = self.criterionGAN(self.pred_B_real, True)
        # Fake
        self.pred_B_fake = self.netD_B(fake_A.detach())     # fake_image.detach()不会回传梯度到G
        loss_D_fake = self.criterionGAN(self.pred_B_fake, False)

        self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_B.backward()

    def backward_G(self):
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # 循环一致loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # 循环一致loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def update_learning_rate(self):
        """每经过一个外部epoch，衰减一次meta_net的lr"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def set_requires_grad(self, nets, requires_grad=False):
        """设置整个模型参数梯度开关"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def get_current_visuals(self):
        """以orderedDict返回图片, 用于可视化"""
        visual_ret = OrderedDict()
        for name in ['real_A', 'fake_B', 'rec_A', 'real_B', 'fake_A', 'rec_B']:
            visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """以orderedDict形式返回loss"""
        lossA_ret = OrderedDict()
        lossB_ret = OrderedDict()
        for name in ['D_A', 'G_A', 'cycle_A']:
            if isinstance(name, str):
                lossA_ret[name] = float(getattr(self, 'loss_' + name))
        for name in ['D_B', 'G_B', 'cycle_B']:
            if isinstance(name, str):
                lossB_ret[name] = float(getattr(self, 'loss_' + name))
        return lossA_ret, lossB_ret

    def get_current_pred(self):
        """以orderedDict形式返回predict"""
        predA_ret = OrderedDict()
        predB_ret = OrderedDict()
        for name in ['A_real', 'A_fake']:
            predA_ret[name] = float(getattr(self, 'pred_' + name).mean())
        for name in ['B_real', 'B_fake']:
            predB_ret[name] = float(getattr(self, 'pred_' + name).mean())
        return predA_ret, predB_ret


