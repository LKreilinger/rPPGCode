import cv2
# internal modules
from Preprocessing.Augmentation import illumination


def augment(img_in):
    # Flip the image vertically
    v_flip = cv2.flip(img_in, 0)
    # Flip the image horizontally
    h_flip = cv2.flip(v_flip, 1)
    # Rotate image 90° clockwise
    r_90 = cv2.rotate(h_flip, cv2.cv2.ROTATE_90_CLOCKWISE)
    # change brightness (darken) and contrast
    contrast = 2
    brightness = -70
    img_out = illumination.adjust_contrast_brightness(r_90, contrast, brightness)
    return img_out