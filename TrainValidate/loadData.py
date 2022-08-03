from kale.loaddata.videos import VideoFrameDataset
from kale.prepdata.video_transform import ImglistToTensor
import os
from torchvision import transforms
import torch



def load_data(outputDataUBFCPath: str, nFramesVideo):
    """
    # Load data as in xdatasetSplit is saved. The split ratio is defined in splitData.py
    :type outputDataUBFCPath: str
    """
    # transformation and uses GPU if possible
    trans = transforms.Compose([
        ImglistToTensor(),
        # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor# [batch,channel,length,width,height] = x.shape
    ])

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    print("GPU available: ",GPU)


    # load train data
    train_Data_path = os.path.join(outputDataUBFCPath, "train")
    train_annotation_file = os.path.join(train_Data_path, "train_annotation.txt")
    train_dataset = VideoFrameDataset(
        root_path=train_Data_path,
        annotationfile_path=train_annotation_file,
        num_segments=1,
        frames_per_segment=nFramesVideo,
        imagefile_template="img_{:05d}.jpg",
        transform=trans,
        random_shift=False,
        test_mode=False

    )
    training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False,
                                                  pin_memory=GPU)

    # load validation data
    val_Data_path = os.path.join(outputDataUBFCPath, "validation")
    val_annotation_file = os.path.join(val_Data_path, "val_annotation.txt")
    val_dataset = VideoFrameDataset(
        root_path=val_Data_path,
        annotationfile_path=val_annotation_file,
        num_segments=1,
        frames_per_segment=nFramesVideo,
        imagefile_template="img_{:05d}.jpg",
        transform=trans,
        random_shift=False,
        test_mode=False
    )
    validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, shuffle=False,
                                                    pin_memory=GPU)

    # load test data
    test_Data_path = os.path.join(outputDataUBFCPath, "test")
    test_annotation_file = os.path.join(test_Data_path, "test_annotation.txt")
    test_dataset = VideoFrameDataset(
        root_path=test_Data_path,
        annotationfile_path=test_annotation_file,
        num_segments=1,
        frames_per_segment=nFramesVideo,
        imagefile_template="img_{:05d}.jpg",
        transform=trans,
        random_shift=False,
        test_mode=True
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False,
                                              pin_memory=GPU)

    return training_loader, validation_loader, test_loader
