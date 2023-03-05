import yaml

from basicsr.data.paired_image_dataset import PairedImageDataset


def test_pairedimagedataset():
    """Test dataset: PairedImageDataset"""

    opt_str = r"""
scale: 4
phase: train
name: RealSR
type: PairedImageDataset
dataroot_gt: datasets/RealSR_V3/All/Train/4_HR_sub.lmdb
dataroot_lq: datasets/RealSR_V3/All/Train/4_LR_sub.lmdb

filename_tmpl: '{}'
io_backend:
#      type: disk
    type: lmdb

gt_size: 192
use_hflip: true
use_rot: true

# data loader
use_shuffle: true
num_worker_per_gpu: 6
batch_size_per_gpu: 32
dataset_enlarge_ratio: 1
prefetch_mode: ~
"""
    opt = yaml.safe_load(opt_str)

    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    print(len(dataset))
    # assert len(dataset) == 2  # whether to read correct meta info

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents

    # ------------------ test filename_tmpl -------------------- #
    opt.pop('filename_tmpl')
    opt['io_backend'] = dict(type='disk')
    dataset = PairedImageDataset(opt)
    assert dataset.filename_tmpl == '{}'
    print(result['lq'].shape,result['gt'].shape)


    dataset = PairedImageDataset(opt)
    assert dataset.io_backend_opt['type'] == 'lmdb'  # io backend
    assert len(dataset) == 2  # whether to read correct meta info
    assert dataset.std == [0.5]

    # test __getitem__
    result = dataset.__getitem__(1)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (1, 128, 128)
    assert result['lq'].shape == (1, 32, 32)
    assert result['lq_path'] == 'comic'
    assert result['gt_path'] == 'comic'

    # ------------------ test case: val/test mode -------------------- #
    opt['phase'] = 'test'
    opt['io_backend'] = dict(type='lmdb')
    dataset = PairedImageDataset(opt)

    # test __getitem__
    result = dataset.__getitem__(0)
    # check returned keys
    expected_keys = ['lq', 'gt', 'lq_path', 'gt_path']
    assert set(expected_keys).issubset(set(result.keys()))
    # check shape and contents
    assert result['gt'].shape == (1, 480, 492)
    assert result['lq'].shape == (1, 120, 123)

if __name__ == '__main__':
    test_pairedimagedataset()