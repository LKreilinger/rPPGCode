import os
import random
import shutil
# internal modules
from Preprocessing import move_s, undo_split

def split_data(config):
    random.seed(1)
    list_subdir = os.listdir(config['genPathData'])
    # check if already solit
    if "test" in list_subdir:
        undo_split.undo_split(config)
    list_subdir = os.listdir(config['genPathData'])

    train_path = os.path.join(config['genPathData'], "train")
    val_path = os.path.join(config['genPathData'], "validation")
    test_path = os.path.join(config['genPathData'], "test")
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)
    # Split in %
    train_split = config['train_split']
    validation_split = config['validation_split']
    test_split = config['test_split']


    number_s = len(list_subdir)
    number_s_train = int(round(train_split / 100 * number_s, 0))
    number_s_val = int(round(validation_split / 100 * number_s, 0))
    number_s_test = int(round(test_split / 100 * number_s, 0))

    # compensate for rounding errors todo compensate with the closest split and not by default train
    if number_s < (number_s_train + number_s_val + number_s_test):
        number_s_train = number_s_train - 1
    if number_s > (number_s_train + number_s_val + number_s_test):
        number_s_train = number_s_train + 1


    list_number_s = list(range(number_s))
    random.shuffle(list_number_s)
    ran_train_list = list_number_s[:number_s_train]
    ran_val_list = list_number_s[number_s_train: (number_s_train + number_s_val)]
    ran_test_list = list_number_s[(number_s_train + number_s_val):]


    move_s.move_s(ran_train_list, config, list_subdir, train_path)
    move_s.move_s(ran_val_list, config, list_subdir, val_path)
    move_s.move_s(ran_test_list, config, list_subdir, test_path)
