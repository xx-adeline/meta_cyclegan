import numpy as np
import visdom
import os
from util.read_write_data import makedir
import time
from util import html, util
from util.read_write_data import write_txt


class Visualizer:
    """用于可视化生成结果、loss以及log"""

    def __init__(self, opt):
        self.opt = opt

        # 连接visdom服务器, plot loss曲线
        self.vis = visdom.Visdom(env="loss and pred")

        # 创建log文件
        log_root = os.path.join(opt.save_path, 'log')
        makedir(log_root)
        # 保存train loss
        self.train_log = os.path.join(self.opt.save_path, 'log/train.log')
        with open(self.train_log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Meta training Loss (%s) ================\n' % now)
        # 保存tune loss
        self.tune_log = os.path.join(self.opt.save_path, 'log/tune.log')
        with open(self.tune_log, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Meta tuning Loss (%s) ================\n' % now)

        # 创建html路径
        self.web_dir = os.path.join(self.opt.save_path, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')

    def plot_line(self, out_epoch, counter_ratio, y, name):
        """
        Parameters:
            epoch (int)
            counter_ratio (float) -- in_epoch跑到当前out_epoch的百分之几
            y (OrderedDict)  -- 待绘制的数据, 比如loss和pred
        """
        if not hasattr(self, 'plot_data'):
            # 待绘制的数据
            plot_data = {'X': [], 'Y': [], 'legend': list(y.keys())}

        # 时间轴
        plot_data['X'].append(out_epoch + counter_ratio)

        # 数值轴，把这一刻的所有y打包成一个list
        plot_data['Y'].append([y[k] for k in plot_data['legend']])

        # 同时画所有y
        self.vis.line(
                      X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
                      Y=np.array(plot_data['Y']),
                      update='append',
                      win=name,
                      opts={
                            'title': name,
                            'legend': plot_data['legend'],
                            'xlabel': 'out epoch',
                            'ylabel': name},
                      )

    def divide_header(self, task_name, is_train=True):
        if is_train:
            write_txt('\n\n---------------- (%s) ----------------' % task_name, self.train_log)
        else:
            write_txt('\n\n---------------- (%s) ----------------' % task_name, self.tune_log)

    def print_current_losses(self, out_epoch, in_epoch, lossA, lossB, is_train=True):
        """打印loss, 并保存在log"""
        if is_train:
            message = "OutEpoch: %d/%d, InEpoch: %d/%d : " % (out_epoch, self.opt.out_epoch_num, in_epoch, self.opt.in_epoch_num)
        else:
            message = "OutEpoch: %d/%d, InEpoch: %d/%d : " % (out_epoch, self.opt.out_epoch_num, in_epoch, self.opt.tune_epoch_num)
        losses = lossA.copy()
        losses.update(lossB)
        for k, v in losses.items():
            message += '%s: %.2f ' % (k, v)
        # 打印该message
        print(message)
        # 保存该message到log中
        if is_train:
            write_txt('%s\n' % message, self.train_log)
        else:
            write_txt('%s\n' % message, self.tune_log)

    def save_images(self, visuals, out_epoch):
        """以html格式保存每轮测试的结果"""

        # 创建html可视化图片
        webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.opt.name, refresh=0)

        # 保存图片
        for label, image in visuals.items():
            image_numpy = util.tensor2im(image)
            img_path = os.path.join(self.img_dir, 'out_epoch%.3d_%s.png' % (out_epoch, label))
            util.save_image(image_numpy, img_path)

        # 制成html形式
        for n in range(out_epoch, 0, -self.opt.test_interval_epoch):
            webpage.add_header('epoch [%d]' % n)
            ims, txts, links = [], [], []

            for label, image in visuals.items():
                img_path = 'out_epoch%.3d_%s.png' % (n, label)
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
            webpage.add_images(ims, txts, links, width=self.opt.crop_size)
        webpage.save()

