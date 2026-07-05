import cv2
import numpy as np


class ImageNormalizer:

    def __init__(self):
        pass

    # --------------------------------------------------
    # Convert to Grayscale
    # --------------------------------------------------
    def grayscale(self, image):

        if len(image.shape) == 2:
            return image

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    # --------------------------------------------------
    # CLAHE Contrast Enhancement
    # --------------------------------------------------
    def clahe(self, gray):

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray)

    # --------------------------------------------------
    # Gamma Correction
    # --------------------------------------------------
    def gamma(self, image, gamma=1.2):

        inv = 1.0 / gamma

        table = np.array([

            ((i / 255.0) ** inv) * 255

            for i in np.arange(256)

        ]).astype("uint8")

        return cv2.LUT(
            image,
            table
        )

    # --------------------------------------------------
    # Denoise
    # --------------------------------------------------
    def denoise(self, image):

        return cv2.fastNlMeansDenoising(
            image,
            None,
            10,
            7,
            21
        )

    # --------------------------------------------------
    # Histogram Equalization
    # --------------------------------------------------
    def equalize(self, gray):

        return cv2.equalizeHist(gray)

    # --------------------------------------------------
    # Gaussian Blur
    # --------------------------------------------------
    def blur(self, image):

        return cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

    # --------------------------------------------------
    # Complete Pipeline
    # --------------------------------------------------
    def normalize(self, image):

        gray = self.grayscale(image)

        gray = self.clahe(gray)

        gray = self.gamma(gray)

        gray = self.denoise(gray)

        gray = self.blur(gray)

        return gray