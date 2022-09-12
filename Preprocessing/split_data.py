import os
import random
import numpy as np
# internal modules
from Preprocessing import move_s, undo_split


def split_data(config):
    random.seed(config['randomSeed'])
    list_subdir = os.listdir(config['genPathData'])
    # check if data already split
    if "train" in list_subdir or "test" in list_subdir or "validation" in list_subdir:
        undo_split.undo_split(config)
    list_subdir = os.listdir(config['genPathData'])
    number_s = len(list_subdir)
    round_error = np.zeros(3)
    number_s_splits = np.zeros(3)

    for idx, key in enumerate(config):
        if config[key] != 0:
            split_path = os.path.join(config['genPathData'], key)
            os.mkdir(split_path)
            number_s_splits[idx] = config[key] / 100 * number_s
            round_error[idx] = number_s_splits[idx] % 1
        if idx == 2:
            break
    # compensate rounding error
    number_s_splits = number_s_splits.astype(int)
    n_zeros = np.count_nonzero(round_error == 0)
    if n_zeros < 2:
        idx_max = round_error.argmax(axis=0)
        number_s_splits[idx_max] += 1

    random.shuffle(list_subdir)
    for idx, key in enumerate(config):
        if config[key] is not 0:
            ran_split_list = list_subdir[:number_s_splits[idx]]
            list_subdir = list_subdir[number_s_splits[idx]:]
            split_path = os.path.join(config['genPathData'], key)
            move_s.move_s(ran_split_list, config, split_path)
        if idx == 2:
            break