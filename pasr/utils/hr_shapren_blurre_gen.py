import numpy as np
import random
import torchvision.transforms.functional as F
import torch
from .gaussian_degrade import SRMDPreprocessing
import argparse

class HR_Transform:
    def __init__(self,opt) -> None:
        # self.blure_size = list(np.arange(1,9,2))
        # self.blure_sigma = list(np.arange(0.01,10,0.05))

        scale = opt.scale
        mode = opt.mode
        kernel_size = opt.kernel_size
        blur_type = opt.blur_type
        sig = opt.sig
        sig_min = opt.sig_min
        sig_max = opt.sig_max
        lambda_1 = opt.lambda_1
        lambda_2 = opt.lambda_2
        theta = opt.theta
        lambda_min = opt.lambda_min
        lambda_max = opt.lambda_max
        noise = opt.noise

        self.degrade = SRMDPreprocessing(
            scale=scale,
            mode = mode,
            kernel_size = kernel_size,
            blur_type=blur_type,
            sig = sig,
            sig_min=sig_min,
            sig_max=sig_max,
            lambda_1 = lambda_1,
            lambda_2 = lambda_2,
            theta = theta,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            noise=noise
        )
        self.gen_num = opt.gen_num
        self.sharpen_scale = list(np.arange(1,5,0.01))

    def add_degerate(self,img,lr):
        imgs = torch.stack([img]*self.gen_num)
        imgs[0] = lr
        for i in range(1,self.gen_num):
            imgs[i],_ = self.degrade(img)
        return imgs
        # kernel_size = (random.sample(self.blure_size,1)[0],random.sample(self.blure_size,1)[0])
        # sigma = (random.sample(self.blure_sigma,1)[0],random.sample(self.blure_sigma,1)[0])
        # img_d = F.gaussian_blur(img,kernel_size=kernel_size,sigma=sigma)
        # return img_d

    def add_sharpen(self,img):
        imgs = torch.stack([img]*self.gen_num)
        for i in range(1,self.gen_num):
            sharpen_rate = random.sample(self.sharpen_scale,1)[0]
            imgs[i] = F.adjust_sharpness(img,sharpen_rate)
        return imgs

    def __call__(self,hr,lr):
        hr_s = self.add_sharpen(hr)
        hr_d = self.add_degerate(hr,lr)
        return {'sharpen':hr_s,'degeratation':hr_d}


def main():
    parser = argparse.ArgumentParser(description='test')
    opt  = parser.parse_args()
    opt.scale = 1
    opt.mode = 'bicubic'
    opt.kernel_size = 21
    opt.blur_type = 'iso_gaussian'
    opt.sig = 0.6
    opt.sig_min = 0.2
    opt.sig_max = 2
    opt.lambda_1 = 0.4
    opt.lambda_2 = 0.8
    opt.theta = 0
    opt.lambda_min = 0.2
    opt.lambda_max = 0.4
    opt.noise = 1
    opt.gen_num = 5

    transfmor = HR_Transform(opt)
    x = torch.rand(4,3,256,256).to('cpu')
    info = transfmor(x)
    s = info['sharpen']
    d = info['degeratation']
    print(s.shape,d.shape)

if __name__ == "__main__":
    main()