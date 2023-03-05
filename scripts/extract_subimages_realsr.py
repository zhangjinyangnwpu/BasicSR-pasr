import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir


def main(gen_scale=2):

    opt = {}
    opt['n_thread'] = 10
    opt['compression_level'] = 3

    if gen_scale == 2:
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/2/HR'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/2/HR_sub'
        opt['crop_size'] = 480
        opt['step'] = 240
        opt['thresh_size'] = 0
        opt['hr'] = True
        extract_subimages(opt)
        # LRx2 images
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/2'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/2_LR_sub'
        opt['crop_size'] = 240
        opt['step'] = 120
        opt['thresh_size'] = 0
        opt['hr'] = False
        extract_subimages(opt)

    elif gen_scale == 3:
        # HR images
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/3'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/3_HR_sub'
        opt['crop_size'] = 480
        opt['step'] = 240
        opt['thresh_size'] = 0
        opt['hr'] = True
        extract_subimages(opt)
        # LRx3 images
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/3'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/3_LR_sub'
        opt['crop_size'] = 160
        opt['step'] = 80
        opt['thresh_size'] = 0
        opt['hr'] = False
        extract_subimages(opt)

    elif gen_scale == 4:
        # HR images
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/4'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/4_HR_sub'
        opt['crop_size'] = 480
        opt['step'] = 240
        opt['thresh_size'] = 0
        opt['hr'] = True
        extract_subimages(opt)
        # LRx4 images
        opt['input_folder'] = 'datasets/RealSR_V3/All/Train/4'
        opt['save_folder'] = 'datasets/RealSR_V3/All/Train/4_LR_sub'
        opt['crop_size'] = 120
        opt['step'] = 60
        opt['thresh_size'] = 0
        opt['hr'] = False
        extract_subimages(opt)
    else:
        raise NotImplemented


def extract_subimages(opt):
    """Crop images to subimages.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print(f'mkdir {save_folder} ...')
    else:
        print(f'Folder {save_folder} already exists. Exit.')
        sys.exit(1)
    img_list = list(scandir(input_folder, full_path=True))
    if opt['hr']:
        img_list = [img_name for img_name in img_list if 'HR' in img_name]
    else:
        img_list = [img_name for img_name in img_list if 'LR' in img_name]
    print(img_list)
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')


def worker(path, opt):
    """Worker for each process.

    Args:
        path (str): Image path.
        opt (dict): Configuration dict. It contains:
            crop_size (int): Crop size.
            step (int): Step for overlapped sliding window.
            thresh_size (int): Threshold size. Patches whose size is lower
                than thresh_size will be dropped.
            save_folder (str): Path to save folder.
            compression_level (int): for cv2.IMWRITE_PNG_COMPRESSION.

    Returns:
        process_info (str): Process information displayed in progress bar.
    """
    crop_size = opt['crop_size']
    step = opt['step']
    thresh_size = opt['thresh_size']
    img_name, extension = osp.splitext(osp.basename(path))

    if opt['hr']:
        img_name = img_name.replace('_HR', '')
    else:
        img_name = img_name.replace('_LR2', '').replace('_LR3', '').replace('_LR4', '')

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    h, w = img.shape[0:2]
    h_space = np.arange(0, h - crop_size + 1, step)
    if h - (h_space[-1] + crop_size) > thresh_size:
        h_space = np.append(h_space, h - crop_size)
    w_space = np.arange(0, w - crop_size + 1, step)
    if w - (w_space[-1] + crop_size) > thresh_size:
        w_space = np.append(w_space, w - crop_size)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            cropped_img = img[x:x + crop_size, y:y + crop_size, ...]
            cropped_img = np.ascontiguousarray(cropped_img)
            cv2.imwrite(
                osp.join(opt['save_folder'], f'{img_name}_s{index:03d}{extension}'), cropped_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    main(gen_scale=4) # gen_scale 表示生成 realsr 的倍数
