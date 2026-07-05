import cv2
import numpy as np


class ImagePreprocessor:

    def __init__(self, max_width=1024, max_height=1024):
        self.max_width = max_width
        self.max_height = max_height

    # ----------------------------------------------------
    # Load Image
    # ----------------------------------------------------
    def load_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        

        return image

    # ----------------------------------------------------
    # Resize
    # ----------------------------------------------------
    def resize_image(self, image):

        h, w = image.shape[:2]

        if w <= self.max_width and h <= self.max_height:
            return image

        scale = min(
            self.max_width / w,
            self.max_height / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        return cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

    # ----------------------------------------------------
    # BGR -> RGB
    # ----------------------------------------------------
    def bgr_to_rgb(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

    # ----------------------------------------------------
    # RGB -> BGR
    # ----------------------------------------------------
    def rgb_to_bgr(self, image):

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

    # ----------------------------------------------------
    # RGB -> YCrCb
    # ----------------------------------------------------
    def convert_to_ycbcr(self, rgb):

        return cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2YCrCb
        )

    # ----------------------------------------------------
    # YCrCb -> RGB
    # ----------------------------------------------------
    def ycbcr_to_rgb(self, ycbcr):

        return cv2.cvtColor(
            ycbcr,
            cv2.COLOR_YCrCb2RGB
        )

    # ----------------------------------------------------
    # Split Channels
    # ----------------------------------------------------
    def split_channels(self, ycbcr):

        y, cr, cb = cv2.split(ycbcr)

        return y, cr, cb

    # ----------------------------------------------------
    # Merge Channels
    # ----------------------------------------------------
    def merge_channels(self, y, cr, cb):

        y = np.clip(y, 0, 255).astype(np.uint8)
        cr = np.clip(cr, 0, 255).astype(np.uint8)
        cb = np.clip(cb, 0, 255).astype(np.uint8)

        return cv2.merge([y, cr, cb])

    # ----------------------------------------------------
    # Normalize
    # ----------------------------------------------------
    def normalize(self, channel):

        return channel.astype(np.float32) / 255.0

    # ----------------------------------------------------
    # Denormalize
    # ----------------------------------------------------
    def denormalize(self, channel):

        channel = channel * 255.0

        channel = np.clip(
            channel,
            0,
            255
        )

        return channel.astype(np.uint8)

    # ----------------------------------------------------
    # Save Image
    # ----------------------------------------------------
    def save_image(self, image, output_path):

        image = self.rgb_to_bgr(image)

        cv2.imwrite(
            output_path,
            image
        )

    # ----------------------------------------------------
    # Embedding Preprocessing
    # ----------------------------------------------------
    def preprocess_for_embedding(self, image_path):

        original = self.load_image(image_path)

        original = self.resize_image(original)

        rgb = self.bgr_to_rgb(original)

        ycbcr = self.convert_to_ycbcr(rgb)

        y, cr, cb = self.split_channels(ycbcr)

        h, w = y.shape

        return {

            "original": original,

            "rgb": rgb,

            "ycbcr": ycbcr,

            "y_channel": y,

            "cr_channel": cr,

            "cb_channel": cb,

            "height": h,

            "width": w
        }

    # ----------------------------------------------------
    # Build Final RGB Image
    # ----------------------------------------------------
    def build_final_image(
        self,
        y,
        cr,
        cb
    ):

        y = np.clip(y, 0, 255).astype(np.uint8)

        merged = self.merge_channels(
            y,
            cr,
            cb
        )

        rgb = self.ycbcr_to_rgb(
            merged
        )

        return rgb

    # ----------------------------------------------------
    # Reconstruction
    # ----------------------------------------------------
    def reconstruct_image(
        self,
        y_channel,
        cr_channel,
        cb_channel
    ):

        return self.build_final_image(
            y_channel,
            cr_channel,
            cb_channel
        )