import os


def pre_config(docker):
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
    return tempPathNofile, genPath, workingPath
