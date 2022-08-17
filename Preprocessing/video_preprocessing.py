import os
from fnmatch import fnmatch
import shutil
# internal modules
from Preprocessing.UBFC import FaceDetection

# %%  Delete dataset folder (inside of output folder)

datasetpath = os.path.join(gen_path, "output", "UBFCDataset")
gen_path_data = os.path.join(gen_path, "data", "UBFC_Phys")
if os.path.exists(datasetpath) and os.path.isdir(datasetpath):
    shutil.rmtree(datasetpath)
os.mkdir(datasetpath)

# Generate temp path for video and BCP-data processing
if docker:
    tempPathNofile = os.path.join(workingPath, "output", "temp")
else:
    tempPathNofile = os.path.join(workingPath, "temp")
if os.path.exists(tempPathNofile) and os.path.isdir(tempPathNofile):
    shutil.rmtree(tempPathNofile)
os.mkdir(tempPathNofile)

# %% Face detection and save images in folder of video name
for path, subdirs, files in os.walk(gen_path_data):
    for name in files:
        currentPath = os.path.join(path, name)
        destinationPath = os.path.join(datasetpath, name)
        tempPath = os.path.join(tempPathNofile, name)
        if fnmatch(name, patternVideo):
            os.mkdir(destinationPath)  # make as many folders as videos excist, Foldername is equal the videoname
            os.makedirs(tempPathNofile, exist_ok=True)
            noFaceList = FaceDetection.viola_jonas_face_detector(currentPath, destinationPath, tempPath,
                                                                 NEW_SAMPLING_RATE, NEW_SIZE_IMAGE)
            n_zeros = np.count_nonzero(noFaceList == 0)
            n_ones = np.count_nonzero(noFaceList == 1)
            shutil.rmtree(tempPathNofile)
            if n_zeros > n_ones:
                nameNoExten = os.path.splitext(name)[0]
                noFaceListAllVideos.append(nameNoExten)
                noFaceListAllVideos.append(noFaceList)
            else:
                shutil.rmtree(destinationPath)
                delete_videos.append(name)