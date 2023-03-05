import math
import torch
from torchvision import models
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.archs.discriminator_arch import VGGStyleDiscriminator
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def smooth_l1_loss(pred, target):
    return F.smooth_l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@LOSS_REGISTRY.register()
class TripleLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(TripleLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.triple_loss = torch.nn.TripletMarginLoss(margin=-1.0,p=2,reduction=self.reduction)

    def forward(self, anchor, pos, neg, weight=None, **kwargs):
        return self.loss_weight * self.triple_loss(anchor, pos, neg)


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


@LOSS_REGISTRY.register()
class ContrastiveLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        extarctor = models.vgg19(pretrained=True).features
        self.extractor = nn.Sequential(
            extarctor,
            nn.AdaptiveAvgPool2d((1,1)),
        ).cuda()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, anchor, pos, neg, weight=None, **kwargs):
        device = anchor.device
        self.label_pos = torch.ones(anchor.shape[0]).to(device)
        self.label_neg = torch.zeros(anchor.shape[0]).to(device)

        with torch.no_grad():
            feature_anchor = self.extractor(anchor)
            feature_pos = self.extractor(pos)
            feature_neg = self.extractor(neg)

            feature_anchor = feature_anchor.view(feature_anchor.shape[0],-1)
            feature_pos = feature_pos.view(feature_pos.shape[0], -1)
            feature_neg = feature_neg.view(feature_neg.shape[0], -1)

        anchor = F.normalize(feature_anchor)
        pos = F.normalize(feature_pos)
        neg = F.normalize(feature_neg)

        sim_pos = F.cosine_similarity(anchor,pos)
        sim_neg = F.cosine_similarity(anchor,neg)

        pos_loss = mse_loss(sim_pos,self.label_pos)
        neg_loss = mse_loss(sim_neg,self.label_neg)

        loss = pos_loss + neg_loss

        del self.label_pos,self.label_neg

        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class ContrastiveLoss_feature(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ContrastiveLoss_feature, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, embedding1, embedding2, is_close = True, norm = True, weight=None, **kwargs):
        device = embedding1.device
        if norm:
            embedding1 = F.normalize(embedding1)
            embedding2 = F.normalize(embedding2)
        if is_close:
            self.label = torch.ones(embedding1.shape[0]).to(device)
        else:
            self.label = torch.zeros(embedding1.shape[0]).to(device)
        similary = F.cosine_similarity(embedding1,embedding2)
        loss = mse_loss(similary,self.label)
        return self.loss_weight * loss


@LOSS_REGISTRY.register()
class ContrastiveLoss_v2(nn.Module):
    def __init__(self,
                 pretrained_path = None,
                 pre_type = 'vgg19',
                 num_in_ch=3,
                 num_feat=64,
                 loss_type='l1',
                 weights=[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0],
                 tempature = 1):
        super(ContrastiveLoss_v2, self).__init__()
        self.pre_type = pre_type
        self.device = 'cpu' if not torch.cuda.is_available() else "cuda"
        if self.pre_type == 'vgg19':
            self.feature_extraction = Vgg19().to(self.device)
        elif self.pre_type == 'esrgan_discriminator':
            self.feature_extraction = VGGStyleDiscriminator(num_in_ch = num_in_ch,num_feat=num_feat)
            if pretrained_path:
                print(pretrained_path)
                self.feature_extraction.load_state_dict(pretrained_path,strict=False)
            self.feature_extraction.to(self.device)
        self.loss_type = loss_type
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.tempature = tempature

    def forward(self, anchor, postive, negative):
        pos_num = postive.shape[0]
        neg_num = negative.shape[0]
        f_postive = [[]]*pos_num
        f_negative = [[]]*neg_num
        if self.pre_type == 'vgg19':
            f_anchor = self.feature_extraction(anchor)
            for i in range(pos_num):
                f_postive[i] = self.feature_extraction(postive[i])
            for i in range(neg_num):
                f_negative[i] = self.feature_extraction(negative[i])
        elif self.pre_type == 'esrgan_discriminator':
            f_anchor = self.feature_extraction.get_feature(anchor)
            for i in range(pos_num):
                f_postive[i] = self.feature_extraction.get_feature(postive[i])
            for i in range(neg_num):
                f_negative[i] = self.feature_extraction.get_feature(negative[i])
        else:
            raise NotImplemented

        if self.loss_type == 'l1':
            loss_func = self.l1_forward
        elif self.loss_type == 'cosine_distance':
            loss_func = self.cosine_forward
        else:
            raise NotImplemented
        loss = loss_func(f_anchor,f_postive,f_negative)
        return loss

    def cosine_forward(self,anchor,postives,negatives):
        pos_num = len(postives)
        neg_num = len(negatives)
        loss = 0
        for i in range(len(anchor)):
            for j in range(pos_num):
                d_ts = torch.cosine_similarity(anchor[i],postives[j][i].detach(),dim=0).mean()
                d_ns_r = d_ts.item()
                for k in range(neg_num):
                    d_ns = torch.cosine_similarity(anchor[i],negatives[k][i].detach(),dim=0).mean()
                    d_ns_r = d_ns_r + d_ns
                contrastive_loss = -torch.log(torch.exp(d_ts/self.tempature)/(torch.exp(d_ns_r/self.tempature)+1e-7))
                contrastive_loss = contrastive_loss.mean(0)
                loss += self.weights[i] * contrastive_loss
        return loss

    def l1_forward(self,anchor,postives,negatives):
        pos_num = len(postives)
        neg_num = len(negatives)
        loss = 0
        for i in range(len(anchor)):
            for j in range(pos_num):
                d_ts = self.l1(anchor[i],postives[j][i].detach())
                d_sn_total = 0
                for k in range(neg_num):
                    d_sn = torch.abs(anchor[i]-negatives[k][i].detach())
                    d_sn = torch.mean(d_ts).sum(0)
                    d_sn_total = d_sn_total + d_sn
                constractive_loss = d_ts / (d_sn_total + 1e-7)
                loss += self.weights[i]*constractive_loss
        return loss