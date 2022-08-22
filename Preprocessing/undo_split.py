import os
import shutil

def undo_split(config):
    list_subdir = os.listdir(config['genPathData'])
    for folder in list_subdir:
        path_subdir = os.path.join(config['genPathData'], folder)
        list_subsubdir = os.listdir(path_subdir)
        for subsub_folder in list_subsubdir:

            src_path = os.path.join(path_subdir, subsub_folder)
            dst_path = os.path.join(config['genPathData'],subsub_folder)
            file_names = os.listdir(src_path)
            os.mkdir(dst_path)
            for file_name in file_names:
                shutil.move(os.path.join(src_path, file_name), os.path.join(dst_path, file_name))
            shutil.rmtree(src_path)
        shutil.rmtree(path_subdir)