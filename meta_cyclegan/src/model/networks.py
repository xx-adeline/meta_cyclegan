import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------tools------------------------------------------------------- #

def get_scheduler(optimizer, opt):
    """保持(out_epoch_num - out_epochs_decay)的初始化lr，经过out_epochs_decay后线性衰减到0"""

    def lambda_rule(epoch):
        lr_weight = 1.0 - max(0, epoch - (opt.out_epoch_num - opt.out_epochs_decay)) / float(opt.out_epochs_decay + 1)
        return lr_weight

    # lr乘以lambda_rule定义的函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """
    Parameters:
        norm_type (str) -- batch | instance | none
    如果是BN,使用可学习的参数，跟踪每个batch的statistics
    如果是IN,使用不可学习参数，不跟踪每个batch的statistics
    """
    if norm_type == 'batch':
        # affine=True, 反归一化的gamma和beta参数可学习
        # track_running_stats=True 跟踪每个batch的statistics，便于test时norm，否则test时计算每个batch实际的statistics
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer  # 都是返回函数


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Parameters:
        net (network)
        init_type (str) -- 初始化方法: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- 自定义方差 for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':  # 正态分布初始化
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':  # 激活层方差递减，保持输入输出方差不变
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':  # 适用于非线性激活层
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':  # 正交
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)  # bias一律初始化为0
        elif classname.find('BatchNorm2d') != -1:
            # 初始化BN的反归一化参数gamma=1，beta=0
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        """
        Parameters:
            target_real_label (bool) - - 定义真为1
            target_fake_label (bool) - - 定义假为0
        """
        super(GANLoss, self).__init__()
        # register_buffer定义需要保存在state_dict的参数，但无需更新梯度
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        """将label广播到与pred相同大小
        Parameters:
            prediction (tensor) - - 判别器的预测概率
            target_is_real (bool) - - ground truth的label
        """

        if target_is_real:
            target_tensor = self.real_label     # 1.0
        else:
            target_tensor = self.fake_label     # 0.0
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """计算判别器预测概率和label的MSE loss
        Parameters:
            prediction (tensor)
            target_is_real (bool)
        """

        # 将label广播到与pred相同大小
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)

        return loss


# ---------------------------------------------------------------------------------------------------------- #
# ----------------------------------------------model------------------------------------------------------- #

def define_G(ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)
    net = ResnetGenerator(3, 3, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_D(ndf, norm='batch', init_type='normal', init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(3, ndf, n_layers=3, norm_layer=norm_layer)
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetGenerator(nn.Module):
    """
    生成器结构
    Perceptual Losses for Real-Time Style Transfer and Super-Resolution
    """

    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect'):
        """
        Parameters:
            input_nc (int)      -- 输入图像channel=3 RGB
            output_nc (int)     -- 生成图像channel=3 RGB
            ngf (int)           -- 最后一个 conv 的 filter 数量
            norm_layer          -- normalization layer
            use_dropout (bool)  -- 是否使用dropout
            n_blocks (int)      -- ResNet blocks的数量
            padding_type (str)  -- reflect | replicate | zero   镜像填充 | 重复填充 | 补零
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        # 加入bias能改善泛化能力，但BN不需要
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # input 1*3*256*256
        # ReflectionPad2d(3), 上下左右pad3,         1*3*262*262
        # Conv2d kernel_size=7, stride=1, (262-7)/1 + 1,    1*64*256*256
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i       # 1, 2
            # (256-3+2)/2 + 1 = 128,    1*128*128*128
            # (128-3+2)/2 + 1 = 64,    1*256*64*64
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # 9层ResNet block
            # channel、size不变    1*256*64*64
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)    # 4, 2
            # 逆卷积   2(63)-2+3+1=128, 1*128*128*128
            #         2(127)-2+3+1=256, 1*64*256*256
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        # 1*64*262*262
        model += [nn.ReflectionPad2d(3)]
        # 1*3*256*256
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        ResNet block
        channel、size不变，两层conv+skip
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- 是否使用dropout
            use_bias (bool)     -- 是否使用bias
        """
        conv_block = []

        # 如果是reflect或replicate, 则加入pad层; 如果是zero padding, 则在conv2d加入
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # channel不变conv
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        # channel不变conv
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        # 加入skip连接
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator(nn.Module):
    """PatchGAN 判别器"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        Parameters:
            input_nc (int)  -- 输入图像的channel
            ndf (int)       -- 最后一层conv的filter数量
            n_layers (int)  -- 判别器的conv数量
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        # 加入bias能改善泛化能力，但BN不需要
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        # input 1*3*256*256
        # (256+2-4)/2 + 1 = 128     1*64*128*128
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # [1:3] 2层
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)    # 2, 4
            # (128+2-4)/2 + 1 = 64      1*128*64*64
            # (64+2-4)/2 + 1 = 32       1*256*32*32
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult  # 4
        nf_mult = min(2 ** n_layers, 8)     # 8
        # (32+2-4)/1 + 1 = 16       1*512*31*31
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # (16+2-4)/1 + 1 = 8    1*1*30*30
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
