from kale.loaddata.videos import VideoFrameDataset
from kale.prepdata.video_transform import ImglistToTensor
import os
from torchvision import transforms
import torch


def load_test_data(config):
    # transformation and uses GPU if possible
    trans = transforms.Compose([
        ImglistToTensor(),
        # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor# [batch,channel,length,width,height] = x.shape
    ])

    if torch.cuda.is_available():
        GPU = True
    else:
        GPU = False
    print("GPU available: ", GPU)

    # load test data
    test_annotation_file = os.path.join(config.path_dataset, "annotations.txt")
    test_dataset = VideoFrameDataset(
        root_path=config.path_dataset,
        annotationfile_path=test_annotation_file,
        num_segments=1,
        frames_per_segment=config.nFramesVideo,
        imagefile_template="img_{:05d}.jpg",
        transform=trans,
        random_shift=False,
        test_mode=True
    )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0, shuffle=False,
                                              pin_memory=GPU)
    return test_loader