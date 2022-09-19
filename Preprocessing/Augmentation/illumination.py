import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")
   return cv2.LUT(image, table)


def adjust_contrast_brightness(img, contrast: float = 1.0, brightness: int = 0):
   """
   Adjusts contrast and brightness of an uint8 image.
   contrast:   (0.0,  inf) with 1.0 leaving the contrast as is
   brightness: [-255, 255] with 0 leaving the brightness as is
   """
   brightness += int(round(255 * (1 - contrast) / 2))
   return cv2.addWeighted(img, contrast, img, 0, brightness)