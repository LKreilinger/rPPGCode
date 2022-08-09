# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 11:15:08 2022

@author: Chaputa
"""

import os
import random
import shutil

def split_data(config):

    # Split in %
    trainSplit = config.train_split
    validationSplit = config.validation_split
    testSplit = config.test_split
    path_dataset = config.path_dataset
    path_dataset_split = config.path_dataset_split

    # new folder of dataset with the subfolders train, validate and test
    if os.path.exists(path_dataset_split) and os.path.isdir(path_dataset_split):
        shutil.rmtree(path_dataset_split)
    os.mkdir(path_dataset_split)
    train_path = os.path.join(path_dataset_split, "train")
    val_path = os.path.join(path_dataset_split, "validation")
    test_path = os.path.join(path_dataset_split, "test")
    os.mkdir(train_path)
    os.mkdir(val_path)
    os.mkdir(test_path)

    # opening the annotation file
    annotation_file = os.path.join(path_dataset, 'annotations.txt')
    full_annotation_file = open(annotation_file, "r")

    # reading the file
    data = full_annotation_file.read()
    full_annotation_file.close()

    # splitting the text when '\n' is seen.
    data_into_list = data.split('\n')

    numberEntries = len(data_into_list)-1 #-1 -> last entry is empty
    numberTrainData = int(round(trainSplit/100*numberEntries,0))
    numberValidationData = int(round(validationSplit/100*numberEntries,0))
    numberTestData = int(round(testSplit/100*numberEntries,0))

    # compensate for rounding errors todo compensate with the closest split and not by default train
    if numberEntries < (numberTrainData + numberValidationData + numberTestData):
        numberTrainData = numberTrainData - 1
    if numberEntries > (numberTrainData + numberValidationData + numberTestData):
        numberTrainData = numberTrainData + 1

    # get for every split an extra annotation file
    ran_list = random.sample(range(numberEntries), numberEntries)
    ran_train_list = ran_list[:numberTrainData]
    ran_val_list = ran_list[numberTrainData: (numberTrainData + numberValidationData)]
    ran_test_list = ran_list[(numberTrainData + numberValidationData):]
    train_annotation = [data_into_list[i] for i in ran_train_list]
    val_annotation = [data_into_list[i] for i in ran_val_list]
    test_annotation = [data_into_list[i] for i in ran_test_list]
    # write annotation list in txt
    train_annotation_path = os.path.join(train_path, "train_annotation.txt")
    val_annotation_path = os.path.join(val_path, "val_annotation.txt")
    test_annotation_path = os.path.join(test_path, "test_annotation.txt")
    f_train = open(train_annotation_path,   "w")
    f_train.write('\n'.join(train_annotation))
    f_train.close()
    f_val = open(val_annotation_path,   "w")
    f_val.write('\n'.join(val_annotation))
    f_val.close()
    f_test = open(test_annotation_path,   "w")
    f_test.write('\n'.join(test_annotation))
    f_test.close()

    # split frames by moving them to new folder
    # generate train set
    print("numberTrainData", numberTrainData)
    for element in range(numberTrainData):
        single_annotation = train_annotation[element:element + 1]
        string_an = ''.join(single_annotation)
        temp_new_folder = string_an[:string_an.index(" ")]
        dst_path = os.path.join(train_path, temp_new_folder)
        print("single_annotation", single_annotation)
        print("string_an", string_an)
        print("temp_new_folder", temp_new_folder)
        print("dst_path", dst_path)
        if not (os.path.exists(dst_path) and os.path.isdir(dst_path)):
            os.mkdir(dst_path)
        string_an = string_an[string_an.find(' ') + 1:]
        st_frame = int(string_an[:string_an.index(" ")])
        string_an = string_an[string_an.find(' ') + 1:]
        end_frame = int(string_an[:string_an.index(" ")])

        src_path = os.path.join(path_dataset, temp_new_folder)
        print("st_frame", st_frame)
        print("end_frame", end_frame)
        print("src_path", src_path)
        for frame in range(st_frame, end_frame + 1):
            iteratImagName = f'img_{frame:05}.jpg'
            src_path_frame = os.path.join(src_path, iteratImagName)
            dst_path_frame = os.path.join(dst_path, iteratImagName)
            print("iteratImagName", iteratImagName)
            print("src_path_frame", src_path_frame)
            print("dst_path_frame", dst_path_frame)
            shutil.move(src_path_frame, dst_path_frame)

    # generate validation set
    for element in range(numberValidationData):
        single_annotation = val_annotation[element:element + 1]
        string_an = ''.join(single_annotation)
        temp_new_folder = string_an[:string_an.index(" ")]
        dst_path = os.path.join(val_path, temp_new_folder)
        if not (os.path.exists(dst_path) and os.path.isdir(dst_path)):
            os.mkdir(dst_path)
        string_an = string_an[string_an.find(' ') + 1:]
        st_frame = int(string_an[:string_an.index(" ")])
        string_an = string_an[string_an.find(' ') + 1:]
        end_frame = int(string_an[:string_an.index(" ")])

        src_path = os.path.join(path_dataset, temp_new_folder)
        for frame in range(st_frame, end_frame + 1):
            iteratImagName = f'img_{frame:05}.jpg'
            src_path_frame = os.path.join(src_path, iteratImagName)
            dst_path_frame = os.path.join(dst_path, iteratImagName)
            shutil.move(src_path_frame, dst_path_frame)

    # generate test set
    for element in range(numberTestData):
        single_annotation = test_annotation[element:element + 1]
        string_an = ''.join(single_annotation)
        temp_new_folder = string_an[:string_an.index(" ")]
        dst_path = os.path.join(test_path, temp_new_folder)
        if not (os.path.exists(dst_path) and os.path.isdir(dst_path)):
            os.mkdir(dst_path)
        string_an = string_an[string_an.find(' ') + 1:]
        st_frame = int(string_an[:string_an.index(" ")])
        string_an = string_an[string_an.find(' ') + 1:]
        end_frame = int(string_an[:string_an.index(" ")])

        src_path = os.path.join(path_dataset, temp_new_folder)
        for frame in range(st_frame, end_frame + 1):
            iteratImagName = f'img_{frame:05}.jpg'
            src_path_frame = os.path.join(src_path, iteratImagName)
            dst_path_frame = os.path.join(dst_path, iteratImagName)
            shutil.move(src_path_frame, dst_path_frame)
