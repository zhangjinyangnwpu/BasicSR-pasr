import torch
from collections import OrderedDict
from os import path as osp

from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

from pasr.utils import SRMDPreprocessing
from pasr.utils import HR_Transform


# add test
@MODEL_REGISTRY.register()
class PASRModel(SRModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(PASRModel, self).__init__(opt)
        self.opt = opt
        self.degrade_flag = self.opt['degrade'].get('flag', None)
        if self.degrade_flag:
            self.degradation = self.init_degrade_generator()
        else:
            self.degradation = None

        self.scale = self.opt['scale']

    def init_degrade_generator(self,):
        scale = self.opt['degrade'].get('scale', None)
        mode = self.opt['degrade'].get('mode', None)
        kernel_size = self.opt['degrade'].get('kernel_size', None)
        blur_type = self.opt['degrade'].get('blur_type', None)
        sig = self.opt['degrade'].get('sig', None)
        sig_min = self.opt['degrade'].get('sig_min', None)
        sig_max = self.opt['degrade'].get('sig_max', None)
        lambda_1 = self.opt['degrade'].get('lambda_1', None)
        lambda_2 = self.opt['degrade'].get('lambda_2', None)
        theta = self.opt['degrade'].get('theta', None)
        lambda_min = self.opt['degrade'].get('lambda_min', None)
        lambda_max = self.opt['degrade'].get('lambda_max', None)
        noise = self.opt['degrade'].get('noise', None)
        gen_num = self.opt['degrade'].get('gen_num', None)

        degrade = SRMDPreprocessing(
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
        class Sample_Gen:
            scale = 1
            mode = None
            kernel_size = None
            blur_type = None
            sig = None
            sig_min = None
            sig_max = None
            lambda_1 = None
            lambda_2 = None
            theta = None
            lambda_min = None
            lambda_max = None
            noise = None
            gen_num = 1

        config = Sample_Gen()
        config.scale = 1
        config.mode = mode
        config.kernel_size = kernel_size
        config.blur_type = blur_type
        config.sig = sig
        config.sig_min = sig_min
        config.sig_max = sig_max
        config.lambda_1 = lambda_1
        config.lambda_2 = lambda_2
        config.theta = theta
        config.lambda_min = lambda_min
        config.lambda_max = lambda_max
        config.noise = noise
        config.gen_num = gen_num

        self.hr_transform = HR_Transform(config)

        return degrade


    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('contrastive_opt'):
            self.cri_contrastive = build_loss(train_opt['contrastive_opt']).to(self.device)
            self.constrastive_weight = train_opt['contrastive_config']['loss_weight']
            self.constrastive_warm_step = train_opt['contrastive_config']['warmup_step']
        else:
            self.cri_contrastive = None

        if train_opt.get('bicubic_loss_opt'):
            self.cri_bicubic = build_loss(train_opt['bicubic_loss_opt']).to(self.device)
            self.bicubic_loss_weight = float(train_opt['bicubic_loss_opt']['loss_weight'])
        else:
            self.cri_bicubic = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        if not self.degrade_flag:
            super().feed_data()
        else:
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                if self.is_train == True:
                    self.lq,_ = self.degradation(self.gt, random = True)
                else:
                    self.lq,_ = self.degradation(self.gt, random = False)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # contrastive loss
        if self.cri_contrastive:
            if self.constrastive_warm_step < current_iter:
                print('aaa',self.scale,self.lq.shape)
                gt_info = self.hr_transform(self.gt,torch.nn.functional.interpolate(self.lq, scale_factor=self.scale, mode='bicubic'))
                gt_shapren = gt_info['sharpen']
                gt_blure = gt_info['degeratation']
                l_contrastive = self.cri_contrastive(self.output,gt_shapren,gt_blure)
                l_total = l_total + l_contrastive * self.constrastive_weight
                loss_dict['l_contrastive'] = l_contrastive
        # bicubic loss
        if self.cri_bicubic:
            bic_sr = torch.nn.functional.interpolate(self.output, scale_factor=1/self.scale, mode='bicubic')
            bic_hr = torch.nn.functional.interpolate(self.gt, scale_factor=1/self.scale, mode='bicubic')
            l_bicbuic_mse = self.cri_bicubic(bic_sr,bic_hr)
            loss_dict['l_bicbuic_mse'] = l_bicbuic_mse
            l_total += l_bicbuic_mse * self.bicubic_loss_weight

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    # 对于尺寸过大的图片，采用切片测试，然后拼接的方式
    def test_split_patch(self):
        print("split patch testing...")
        args = {
            "tile": 512,
            "tile_overlap": 0,
            "scale": self.scale,
        }
        window_size = 256
        b, c, h, w = self.lq.size()
        print( b, c, h, w)
        tile = min(args["tile"], h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        tile_overlap = args["tile_overlap"]
        sf = args["scale"]

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*sf, w*sf).type_as(self.lq)
        W = torch.zeros_like(E)

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output = self.net_g_ema(self.lq)
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = self.lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        b,c,w,h = in_patch.shape
                        out_patch = self.net_g_ema(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)
                        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
        else:
            self.net_g.eval()
            with torch.no_grad():
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        in_patch = self.lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                        b,c,w,h = in_patch.shape
                        out_patch = self.net_g(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)
                        E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch)
                        W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask)
            self.net_g.train()
        self.output = E.float().div_(W.float())