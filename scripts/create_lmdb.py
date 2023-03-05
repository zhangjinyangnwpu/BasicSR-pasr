import argparse
from os import path as osp

from basicsr.utils import scandir
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

# tip 未检查路径
def create_lmdb_for_div2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K/DIV2K_train_HR_sub'
    lmdb_path = 'datasets/DIV2K/DIV2K_train_HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True,n_thread=16)

    # LRx2 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx3 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X3_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx4 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx8 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X8_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X8_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def create_lmdb_for_div2k_flickr2k():
    """Create lmdb files for DIV2K dataset.

    Usage:
        Before run this script, please run `extract_subimages.py`.
        Typically, there are four folders to be processed for DIV2K dataset.
            DIV2K_train_HR_sub
            DIV2K_train_LR_bicubic/X2_sub
            DIV2K_train_LR_bicubic/X3_sub
            DIV2K_train_LR_bicubic/X4_sub
        Remember to modify opt configurations according to your settings.
    """
    # HR images
    folder_path = 'datasets/DIV2K_Flickr2K/HR_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/HR_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx2 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X2_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X2_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx3 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X3_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X3_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx4 images
    folder_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X4_sub'
    lmdb_path = 'datasets/DIV2K_Flickr2K/LR_bicubic/X4_sub.lmdb'
    img_path_list, keys = prepare_keys_div2k(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # LRx8 images
    # folder_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic/X8_sub'
    # lmdb_path = 'datasets/DIV2K/DIV2K_train_LR_bicubic_X8_sub.lmdb'
    # img_path_list, keys = prepare_keys_div2k(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys



def create_lmdb_for_realsr(gen_scale=2):
    if gen_scale == 2:
        # HR images  change dataset path
        folder_path = 'datasets/RealSR_V3/All/Train/2_HR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/2_HR_sub.lmdb'
        img_path_list, keys = prepare_keys_realsr(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

        # LRx2 images
        folder_path = 'datasets/RealSR_V3/All/Train/2_LR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/2_LR_sub.lmdb'
        img_path_list, keys = prepare_keys_div2k(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    elif gen_scale == 3:
        # HR images  change dataset path
        folder_path = 'datasets/RealSR_V3/All/Train/3_HR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/3_HR_sub.lmdb'
        img_path_list, keys = prepare_keys_realsr(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

        # LRx3 images
        folder_path = 'datasets/RealSR_V3/All/Train/3/LR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/3_LR_sub.lmdb'
        img_path_list, keys = prepare_keys_div2k(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    elif gen_scale == 4:
        # HR images  change dataset path
        folder_path = 'datasets/RealSR_V3/All/Train/4_HR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/4_HR_sub.lmdb'
        img_path_list, keys = prepare_keys_realsr(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
        # LRx4 images
        folder_path = 'datasets/RealSR_V3/All/Train/4_LR_sub'
        lmdb_path = 'datasets/RealSR_V3/All/Train/4_LR_sub.lmdb'
        img_path_list, keys = prepare_keys_realsr(folder_path)
        make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    else:
        raise NotImplemented


def prepare_keys_realsr(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

# tip 未检查路径
def create_lmdb_for_drealsr():
    # LRx2 images
    # # HR x2 images  change dataset path
    folder_path = 'datasets/DRealSR/x2/Train_x2/train_HR'
    lmdb_path = 'datasets/DRealSR/x2/Train_x2/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx2 images
    folder_path = 'datasets/DRealSR/x2/Train_x2/train_LR'
    lmdb_path = 'datasets/DRealSR/x2/Train_x2/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # # HR x3 images  change dataset path
    folder_path = 'datasets/DRealSR/x3/Train_x3/train_HR'
    lmdb_path = 'datasets/DRealSR/x3/Train_x3/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx3 images
    folder_path = 'datasets/DRealSR/x3/Train_x3/train_LR'
    lmdb_path = 'datasets/DRealSR/x3/Train_x3/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)

    # # HR x4 images  change dataset path
    folder_path = 'datasets/DRealSR/x4/Train_x4/train_HR'
    lmdb_path = 'datasets/DRealSR/x4/Train_x4/train_HR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)
    #
    # # LRx4 images
    folder_path = 'datasets/DRealSR/x4/Train_x4/train_LR'
    lmdb_path = 'datasets/DRealSR/x4/Train_x4/train_LR.lmdb'
    img_path_list, keys = prepare_keys_drealsr(folder_path)
    # make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys,multiprocessing_read=True)




def prepare_keys_drealsr(folder_path):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='div2k',
        help=("Options: 'realsr', 'drealsr' "
              'You may need to modify the corresponding configurations in codes.'))
    parser.add_argument(
        '--gen_scale',
        type=int,
        default=4,
        help="Options: 2 3 4")
    args = parser.parse_args()
    dataset = args.dataset.lower()
    gen_scale = args.gen_scale
    if dataset == 'div2k':
        create_lmdb_for_div2k(gen_scale=gen_scale)
    elif dataset == 'div2k_flickr2k':
        create_lmdb_for_div2k_flickr2k(gen_scale=gen_scale)
    elif dataset == 'realsr':
        create_lmdb_for_realsr(gen_scale=gen_scale)
    elif dataset == 'drealsr':
        create_lmdb_for_drealsr(gen_scale=gen_scale)
    else:
        raise ValueError('Wrong dataset.')
