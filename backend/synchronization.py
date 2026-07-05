import cv2
import numpy as np


class Synchronizer:

    def __init__(self):
        pass

    # --------------------------------------------------
    # Center Crop
    # --------------------------------------------------
    def center_crop(self, image, width, height):

        h, w = image.shape[:2]

        start_x = max((w - width) // 2, 0)
        start_y = max((h - height) // 2, 0)

        end_x = start_x + width
        end_y = start_y + height

        return image[start_y:end_y, start_x:end_x]

    # --------------------------------------------------
    # Resize
    # --------------------------------------------------
    def resize(self, image, width, height):

        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------
    # Align Size
    # --------------------------------------------------
    def align(self, image, target_shape):

        height, width = target_shape

        return self.resize(
            image,
            width,
            height
        )

    # --------------------------------------------------
    # Pad Image
    # --------------------------------------------------
    def pad(self, image, target_shape):

        height, width = target_shape

        h, w = image.shape[:2]

        top = max((height - h) // 2, 0)
        bottom = max(height - h - top, 0)

        left = max((width - w) // 2, 0)
        right = max(width - w - left, 0)

        return cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=255
        )

    # --------------------------------------------------
    # Synchronize
    # --------------------------------------------------
    def synchronize(self, image, target_shape):

        image = self.align(
            image,
            target_shape
        )

        return image

    # --------------------------------------------------
    # Image Information
    # --------------------------------------------------
    def info(self, image):

        return {

            "height": image.shape[0],

            "width": image.shape[1]

        }