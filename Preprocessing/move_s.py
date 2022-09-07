import os
import shutil

def move_s(ran_list, config, split_path):
    for s in ran_list:
        src_path = os.path.join(config['genPathData'], s)
        dst_path = os.path.join(split_path, s)
        file_names = os.listdir(src_path)
        os.mkdir(dst_path)
        for file_name in file_names:
            shutil.move(os.path.join(src_path, file_name), os.path.join(dst_path, file_name))
        shutil.rmtree(src_path)