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
        # 外部模型 load CPU
        self.netG_A = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        self.netG_B = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        self.netD_A = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain)
        self.netD_B = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain)

        # 内部模型 load GPU
        self.netG_A_meta = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain).to(opt.device)
        self.netG_B_meta = networks.define_G(opt.ngf, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain).to(opt.device)

        self.netD_A_meta = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain).to(opt.device)
        self.netD_B_meta = networks.define_D(opt.ndf, opt.norm, opt.init_type, opt.init_gain).to(opt.device)

        # 判别器以50%概率使用历史生成图像，以50%概率使用当前生成图像
        self.meta_fake_A_pool = ImagePool(opt.pool_size)
        self.meta_fake_B_pool = ImagePool(opt.pool_size)

        # 创建损失函数
        self.criterionGAN = networks.GANLoss().to(opt.device)
        self.criterionCycle = torch.nn.L1Loss().to(opt.device)

        # 外部优化器
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.out_lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.out_lr, betas=(opt.beta1, 0.999))

        # 内部优化器
        self.optimizer_G_meta = torch.optim.Adam(itertools.chain(self.netG_A_meta.parameters(), self.netG_B_meta.parameters()), lr=opt.in_lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_meta = torch.optim.Adam(itertools.chain(self.netD_A_meta.parameters(), self.netD_B_meta.parameters()), lr=opt.in_lr, betas=(opt.beta1, 0.999))

        # 给内部模型的每个优化器, 创建学习率衰减器
        self.optimizers_meta = []
        self.optimizers_meta.append(self.optimizer_G_meta)
        self.optimizers_meta.append(self.optimizer_D_meta)
        self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers_meta]

    def meta_forward(self, img_A, img_B):
        self.real_A = img_A
        self.real_B = img_B

        self.fake_B = self.netG_A_meta(self.real_A)      # G_A(A)
        self.rec_A = self.netG_B_meta(self.fake_B)       # G_B(G_A(A))
        self.fake_A = self.netG_B_meta(self.real_B)      # G_B(B)
        self.rec_B = self.netG_A_meta(self.fake_A)       # G_A(G_B(B))

    def meta_optimize_parameters(self, img_A, img_B):
        """优化内部模型参数"""
        # forward
        self.meta_forward(img_A, img_B)
        # 关闭内部模型D的梯度
        self.set_requires_grad([self.netD_A_meta, self.netD_B_meta], False)
        # 计算内部模型G的loss
        self.optimizer_G_meta.zero_grad()
        self.meta_backward_G()
        self.optimizer_G_meta.step()
        # 暂存可视化G_pred的数据
        self.meta_G_pA_ret, self.meta_G_pB_ret = self.get_current_pred()
        # 打开内部模型D的梯度    fake_image.detach()不会回传梯度到G
        self.set_requires_grad([self.netD_A_meta, self.netD_B_meta], True)
        # 计算内部模型D的loss
        self.optimizer_D_meta.zero_grad()
        self.meta_backward_D_A()
        self.meta_backward_D_B()
        self.optimizer_D_meta.step()
        # 暂存可视化D_pred的数据
        self.meta_D_pA_ret, self.meta_D_pB_ret = self.get_current_pred()

    def meta_backward_D_A(self):
        """内部模型判别器A的backward"""
        fake_B = self.meta_fake_B_pool.query(self.fake_B)
        # Real
        self.pred_A_real = self.netD_A_meta(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_A_real, True)
        # Fake
        self.pred_A_fake = self.netD_A_meta(fake_B.detach())     # fake_image.detach()不会回传梯度到G
        loss_D_fake = self.criterionGAN(self.pred_A_fake, False)

        self.loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_A.backward()

    def meta_backward_D_B(self):
        """内部模型判别器B的backward"""
        fake_A = self.meta_fake_A_pool.query(self.fake_A)
        # Real
        self.pred_B_real = self.netD_B_meta(self.real_A)
        loss_D_real = self.criterionGAN(self.pred_B_real, True)
        # Fake
        self.pred_B_fake = self.netD_B_meta(fake_A.detach())     # fake_image.detach()不会回传梯度到G
        loss_D_fake = self.criterionGAN(self.pred_B_fake, False)

        self.loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        self.loss_D_B.backward()

    def meta_backward_G(self):
        """内部模型生成器AB的backward"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # 以生成器视角的可视化
        self.pred_A_fake = self.netD_A_meta(self.fake_B)
        self.pred_B_fake = self.netD_B_meta(self.fake_A)
        if self.opt.g_pred:
            with torch.no_grad():
                self.pred_A_real = self.netD_A_meta(self.real_B)
                self.pred_B_real = self.netD_B_meta(self.real_A)

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.pred_A_fake, True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.pred_B_fake, True)
        # 循环一致loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # 循环一致loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def update_learning_rate(self):
        """每经过一个out epoch，衰减一次内部模型的lr"""
        old_lr = self.optimizers_meta[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers_meta[0].param_groups[0]['lr']
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

