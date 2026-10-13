import os


def pre_config(docker, base_dir=None):
    # for docker change workdir
    if docker:
        print("Docker is working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = workingPath
        tempPathNofile = os.path.join(workingPath, "output", "temp")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        print("Docker is NOT working")
        workingPath = os.path.abspath(os.getcwd())
        genPath = os.path.dirname(workingPath)
        tempPathNofile = os.path.join(workingPath, "temp")
    # folder that contains the data and the output folder
    if base_dir:
        genPath = os.path.abspath(base_dir)
    print("Data folder:  ", os.path.join(genPath, "data"))
    print("Output folder:", os.path.join(genPath, "output"))
    return tempPathNofile, genPath, workingPath
