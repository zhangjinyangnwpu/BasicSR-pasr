import sys,os
sys.path.append("..")

import yaml
import tempfile
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

import torch
from basicsr.losses.basic_loss import L1Loss
from basicsr.data.paired_image_dataset import PairedImageDataset
from pasr.models.pasr_model import PASRModel
from pasr.archs.pasr_arch import PASR
from pasr.losses.contrastive_loss import ContrastiveLoss_v2


def test_pasr_model():
    opt_str = r"""
scale: 4
num_gpu: 0
manual_seed: 0
is_train: True
dist: False
# network structures
network_g:
    type: PASR
    input_channels: 3
    output_channels: 3
    scale: 4
    num_layers: 5
    use_squeeze: False
    fea_dim: 32
# path
path:
    pretrain_network_g: ~
    strict_load_g: true
    resume_state: ~
# training settings
train:
    ema_decay: 0.999
    optim_g:
        type: Adam
        lr: !!float 2e-4
        weight_decay: 0
        betas: [0.9, 0.99]
    scheduler:
        type: CosineAnnealingRestartLR
        periods: [250000, 250000, 250000, 250000]
        restart_weights: [1, 1, 1, 1]
        eta_min: !!float 1e-7
    total_iter: 1000000
    warmup_iter: -1  # no warm up
    # losses
    pixel_opt:
        type: L1Loss
        loss_weight: 1.0
        reduction: mean
    contrastive_opt:
        type: ContrastiveLoss_v2
    contrastive_config:
        loss_weight: 1.0
        warmup_step: 0
    bicubic_loss_opt:
        type: L1Loss
        loss_weight: 1.0
        reduction: mean


# for gaussion degrade
degrade:
    flag: True
    scale: 4
    mode: bicubic
    kernel_size: 21 # gaussian kernel size
    blur_type: aniso_gaussian # iso_gaussian or aniso_gaussian
    sig: 0.6       # test with a certain value for iso_gaussian
    sig_min: 0.2   # training 0.2 for x2, 0.2 for x3, 0.2 for x4 for iso_gaussian
    sig_max: 2.0   # training 2.0 for x2, 3.0 for x2, 4.0 for x4 for iso_gaussian
    lambda_1: 1.0  # test with a cetrain value for aniso_gaussian
    lambda_2: 2.0  # test with a cetrain value for aniso_gaussian
    theta: 0       # angle for aniso_gaussian, set with angle when testing
    lambda_min: 0.2 # training 0.2 for x2,x3,x4 for aniso_gaussian
    lambda_max: 4.0 # training 4.0 for x2,x3,x4 for aniso_gaussian
    noise: 10 # random for training and testing for valiation
    gen_num: 1

# validation settings
val:
    val_freq: !!float 5e3
    save_img: True

    metrics:
        psnr: # metric name
            type: calculate_psnr
            crop_border: 4
            test_y_channel: false
            better: higher  # the higher, the better. Default: higher
        ssim:
            type: calculate_ssim
            crop_border: 4
            test_y_channel: false
            better: higher  # the higher, the better. Default: higher
        niqe:
            type: calculate_niqe
            crop_border: 4
            test_y_channel: false
            better: higher  # the higher, the better. Default: higher
"""
    opt = yaml.safe_load(opt_str)

    # build model
    model = PASRModel(opt)
    assert model.__class__.__name__ == 'PASRModel'
    assert isinstance(model.net_g, PASR)
    assert isinstance(model.cri_pix, L1Loss)
    # 待完善
    assert isinstance(model.cri_contrastive, ContrastiveLoss_v2)
    assert isinstance(model.cri_bicubic, L1Loss)

    assert isinstance(model.optimizers[0], torch.optim.Adam)
    assert model.ema_decay == 0.999
    gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    lq = torch.rand((1, 3, 8, 8), dtype=torch.float32)
    sr = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    data = dict(gt=gt, lq=lq)
    model.feed_data(data)

    model.optimize_parameters(1)
    assert model.output.shape == (1, 3, 32, 32)
    assert isinstance(model.log_dict, dict)

    # ----------------- test loss function -------------------- #
    lr_inter = torch.nn.functional.interpolate(lq,scale_factor=4)
    bic_sr = torch.nn.functional.interpolate(sr, scale_factor=1/4, mode='bicubic')
    bic_hr = torch.nn.functional.interpolate(gt, scale_factor=1/4, mode='bicubic')
    print(f"lr_inter:{lr_inter.shape}")
    loss_pixel = model.cri_pix(gt,sr)
    print(gt.shape,sr.shape)
    loss_contrastic = model.cri_contrastive(sr,gt,lr_inter)
    loss_bicubic = model.cri_bicubic(bic_sr,bic_hr)
    print(f"loss_pixel:{loss_pixel}")
    print(f"loss_contrastic:{loss_contrastic}")
    print(f"loss_bicubic:{loss_bicubic}")

    # ----------------- test save -------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['models'] = tmpdir
        model.opt['path']['training_states'] = tmpdir
        model.save(0, 1)

    # ----------------- test the test function -------------------- #
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    # delete net_g_ema
    model.__delattr__('net_g_ema')
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    assert model.net_g.training is True  # should back to training mode after testing

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = dict(
        name='test',
        dataroot_gt='../datasets/Classical/Set5/GTmod12',
        dataroot_lq='../datasets/Classical/Set5/LRbicx4',
        io_backend=dict(type='disk'),
        scale=4,
        phase='val')
    dataset = PairedImageDataset(dataset_opt)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt['path']['visualization'] = tmpdir
        model.nondist_validation(dataloader, 1, None, save_img=True)
        assert model.is_train is True
        # check metric_results
        assert 'psnr' in model.metric_results
        assert isinstance(model.metric_results['psnr'], float)

    for idx,data in enumerate(dataloader):
        lq = data['lq']
        gt = data['gt']
        lq_d,k = model.degradation(gt,random=True)
        print(lq_d.shape, k.shape)
        transform = T.ToPILImage()
        img = transform(lq_d[0])
        # kernel = transform(k)
        img.save(f"../tmp/{idx}_img.png")
        # kernel.save(f"../tmp/{idx}_kernel.png")
        kernel = k[0].numpy()
        plt.axis('off')
        plt.imshow(kernel,cmap='binary_r',interpolation='bicubic')# binary_r binary
        plt.savefig(f"../tmp/{idx}_kernel.png",bbox_inches='tight', pad_inches=0)

    # # in validation mode
    # with tempfile.TemporaryDirectory() as tmpdir:
    #     model.opt['is_train'] = False
    #     model.opt['val']['suffix'] = 'test'
    #     model.opt['path']['visualization'] = tmpdir
    #     model.opt['val']['pbar'] = False
    #     model.nondist_validation(dataloader, 1, None, save_img=True)
    #     # check metric_results
    #     assert 'psnr' in model.metric_results
    #     assert isinstance(model.metric_results['psnr'], float)

    #     # if opt['val']['suffix'] is None
    #     model.opt['val']['suffix'] = None
    #     model.opt['name'] = 'demo'
    #     model.opt['path']['visualization'] = tmpdir
    #     model.nondist_validation(dataloader, 1, None, save_img=True)
    #     # check metric_results
    #     assert 'psnr' in model.metric_results
    #     assert isinstance(model.metric_results['psnr'], float)
    #     print(model.metric_results)

if __name__ == '__main__':
    os.makedirs('../tmp',exist_ok=True)
    test_pasr_model()