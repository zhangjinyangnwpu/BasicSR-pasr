# flake8: noqa

import os.path as osp
from basicsr.train import train_pipeline

import pasr.losses
import pasr.archs
import pasr.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)