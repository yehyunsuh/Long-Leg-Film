import os
import argparse
import ast
import torch
import random
import numpy as np

from glob import glob


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Arguemtn \"%s\" is not a list" % (s))
    return v


def customize_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def png_to_txt():
    if os.path.isfile('./data/txt/test_label.txt'):
        os.remove('./data/txt/test_label.txt')
    os.makedirs('./data/txt', exist_ok=True)

    png_list = glob('./data/image/Both_test_set_EN_cropped/*.png')
    for png in png_list:
        image_name = f'{png.split("/")[-1]}\n'
        file_write = open('./data/txt/test_label.txt', "a+")
        file_write.write(image_name)
        file_write.close()