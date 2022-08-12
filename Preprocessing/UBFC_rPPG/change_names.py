import os
from fnmatch import fnmatch
path = r"J:\Masterarbeit"

for path, subdirs, files in os.walk(path):
    print(files)
    if not(files==[]):
        for file in files:
            if file[0] == 'b':
                source = os.path.join(path, file)
                num = ""
                for c in path:
                    if c.isdigit():
                        num = num + c
                dest = os.path.join(path, "bvp_s" + num + ".txt")
                os.rename(source, dest)
            if file[0] == 'v':
                source = os.path.join(path, file)
                num = ""
                for c in path:
                    if c.isdigit():
                        num = num + c
                dest = os.path.join(path, "vid_s" + num + ".avi")
                os.rename(source, dest)


    for s_path in subdirs:
        source = os.path.join(path, s_path)
        num = ""
        for c in s_path:
            if c.isdigit():
                num = num + c
        dest = os.path.join(path, "s" + num)
        os.rename(source, dest)