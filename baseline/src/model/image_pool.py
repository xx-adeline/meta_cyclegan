import random
import torch


# 缓存历史生成图像
class ImagePool:

    def __init__(self, pool_size):
        """
        pool_size (int) -- 图像缓冲区的大小，如果pool_size=0，则不会创建缓冲区
        """
        self.pool_size = pool_size
        # 创建空buffer
        if self.pool_size > 0:
            self.num_imgs = 0   # 当前buffer存储的image数量
            self.images = []    # 列表作buffer

    def query(self, images):
        """
        images: 当前的生成图片
        """
        # 如果没创建pool，则返回input
        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            # 如果buffer没满，则向buffer插入当前生成图片，同时输出给判别器
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)

            # 如果buffer满了
            else:
                p = random.uniform(0, 1)
                # 50%的概率，随机取一个buffer存储的image输出给判别器,并向buffer插入当前生成图片
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                # 50%的概率，buffer不变，输出当前生成图片给判别器
                else:
                    return_images.append(image)

        return_images = torch.cat(return_images, 0)
        return return_images
