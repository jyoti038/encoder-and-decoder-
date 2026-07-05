import pywt
import numpy as np
import cv2


class DWTProcessor:
    """
    DWT Processor
    -------------
    Handles Forward and Inverse Discrete Wavelet Transform.
    """

    def __init__(self, wavelet="haar", level=1):
        self.wavelet = wavelet
        self.level = level

    # ----------------------------------------------------
    # Validate Input
    # ----------------------------------------------------
    def validate_channel(self, channel):

        if channel is None:
            raise ValueError("Input channel is None.")

        if not isinstance(channel, np.ndarray):
            raise TypeError("Input must be numpy array.")

        if channel.ndim != 2:
            raise ValueError("Input must be grayscale.")

        return True

    # ----------------------------------------------------
    # Padding
    # ----------------------------------------------------
    def pad_image(self, channel):

        rows, cols = channel.shape

        pad_r = rows % 2
        pad_c = cols % 2

        if pad_r != 0 or pad_c != 0:

            channel = np.pad(
                channel,
                (
                    (0, pad_r),
                    (0, pad_c)
                ),
                mode="edge"
            )

        return channel

    # ----------------------------------------------------
    # Remove Padding
    # ----------------------------------------------------
    def remove_padding(self, image, original_shape):

        h, w = original_shape

        return image[:h, :w]

    # ----------------------------------------------------
    # Forward DWT
    # ----------------------------------------------------
    def forward(self, channel):

        self.validate_channel(channel)

        original_shape = channel.shape

        channel = self.pad_image(channel)

        coeffs = pywt.dwt2(
            channel,
            self.wavelet
        )

        LL, (LH, HL, HH) = coeffs

        return {
            "LL": LL,
            "LH": LH,
            "HL": HL,
            "HH": HH,
            "shape": original_shape
        }

    # ----------------------------------------------------
    # Inverse DWT
    # ----------------------------------------------------
    def inverse(self, coeff_dict):

        coeffs = (
            coeff_dict["LL"],
            (
                coeff_dict["LH"],
                coeff_dict["HL"],
                coeff_dict["HH"]
            )
        )

        reconstructed = pywt.idwt2(
            coeffs,
            self.wavelet
        )

        reconstructed = self.remove_padding(
            reconstructed,
            coeff_dict["shape"]
        )

        return reconstructed
    
        # ----------------------------------------------------
    # Get Subband
    # ----------------------------------------------------
    def get_band(self, coeff_dict, band):

        band = band.upper()

        if band not in ["LL", "LH", "HL", "HH"]:
            raise ValueError("Invalid DWT band.")

        return coeff_dict[band]

    # ----------------------------------------------------
    # Replace Subband
    # ----------------------------------------------------
    def replace_band(self, coeff_dict, band, values):

        band = band.upper()

        if band not in ["LL", "LH", "HL", "HH"]:
            raise ValueError("Invalid DWT band.")

        coeff_dict[band] = values

        return coeff_dict

    # ----------------------------------------------------
    # Band Statistics
    # ----------------------------------------------------
    def statistics(self, coeff_dict):

        stats = {}

        for band in ["LL", "LH", "HL", "HH"]:

            data = coeff_dict[band]

            stats[band] = {

                "shape": data.shape,

                "min": float(np.min(data)),

                "max": float(np.max(data)),

                "mean": float(np.mean(data)),

                "std": float(np.std(data))
            }

        return stats

    # ----------------------------------------------------
    # Normalize Band
    # ----------------------------------------------------
    def normalize_band(self, band):

        minimum = np.min(band)

        maximum = np.max(band)

        if maximum == minimum:

            return band

        return (band - minimum) / (maximum - minimum)

            # ----------------------------------------------------
    # Multi Level Forward DWT
    # ----------------------------------------------------
    def forward_multilevel(self, channel):

        self.validate_channel(channel)

        original_shape = channel.shape

        channel = self.pad_image(channel)

        coeffs = pywt.wavedec2(
            channel,
            wavelet=self.wavelet,
            level=self.level
        )

        return {
            "coeffs": coeffs,
            "shape": original_shape
        }

    # ----------------------------------------------------
    # Multi Level Inverse DWT
    # ----------------------------------------------------
    def inverse_multilevel(self, coeff_dict):

        reconstructed = pywt.waverec2(
            coeff_dict["coeffs"],
            self.wavelet
        )

        reconstructed = self.remove_padding(
            reconstructed,
            coeff_dict["shape"]
        )

        return reconstructed

    # ----------------------------------------------------
    # Capacity
    # ----------------------------------------------------
    def calculate_capacity(self, coeff_dict, band="HL"):

        band = band.upper()

        if band not in coeff_dict:
            raise ValueError("Invalid Band")

        h, w = coeff_dict[band].shape

        return h * w

    # ----------------------------------------------------
    # Band Energy
    # ----------------------------------------------------
    def band_energy(self, band):

        return float(np.sum(np.square(band)))

    # ----------------------------------------------------
    # Energy Report
    # ----------------------------------------------------
    def energy_report(self, coeff_dict):

        report = {}

        for b in ["LL", "LH", "HL", "HH"]:

            report[b] = self.band_energy(
                coeff_dict[b]
            )

        return report

    # ----------------------------------------------------
    # Best Band Selection
    # ----------------------------------------------------
    def best_embedding_band(self, coeff_dict):

        report = self.energy_report(coeff_dict)

        report.pop("LL")

        best = max(report, key=report.get)

        return best

    # ----------------------------------------------------
    # Clip Coefficients
    # ----------------------------------------------------
    def clip_coefficients(self, coeff_dict):

        for b in ["LL", "LH", "HL", "HH"]:

            coeff_dict[b] = np.nan_to_num(
                coeff_dict[b]
            )

        return coeff_dict

    # ----------------------------------------------------
    # Copy Coefficients
    # ----------------------------------------------------
    def clone(self, coeff_dict):

        copied = {}

        for key in coeff_dict:

            if isinstance(coeff_dict[key], np.ndarray):

                copied[key] = coeff_dict[key].copy()

            else:

                copied[key] = coeff_dict[key]

        return copied

            # ----------------------------------------------------
    # Shannon Entropy
    # ----------------------------------------------------
    def calculate_entropy(self, band):

        band = band.astype(np.float32)

        band = cv2.normalize(
            band,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        hist = cv2.calcHist(
            [band],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist = hist.ravel()

        hist = hist / np.sum(hist)

        hist = hist[hist > 0]

        entropy = -np.sum(
            hist * np.log2(hist)
        )

        return float(entropy)

    # ----------------------------------------------------
    # Block Split
    # ----------------------------------------------------
    def split_into_blocks(
        self,
        band,
        block_size=8
    ):

        h, w = band.shape

        blocks = []

        for i in range(0, h, block_size):

            for j in range(0, w, block_size):

                block = band[
                    i:i+block_size,
                    j:j+block_size
                ]

                if block.shape == (
                    block_size,
                    block_size
                ):

                    blocks.append({

                        "row": i,

                        "col": j,

                        "block": block

                    })

        return blocks

    # ----------------------------------------------------
    # Merge Blocks
    # ----------------------------------------------------
    def merge_blocks(
        self,
        blocks,
        shape,
        block_size=8
    ):

        image = np.zeros(
            shape,
            dtype=np.float32
        )

        for item in blocks:

            r = item["row"]

            c = item["col"]

            image[
                r:r+block_size,
                c:c+block_size
            ] = item["block"]

        return image

            # ----------------------------------------------------
    # Block Energy
    # ----------------------------------------------------
    def block_energy(self, block):

        return np.sum(
            np.square(block)
        )

    # ----------------------------------------------------
    # Rank Blocks
    # ----------------------------------------------------
    def rank_blocks(
        self,
        blocks
    ):

        ranked = []

        for item in blocks:

            energy = self.block_energy(
                item["block"]
            )

            ranked.append({

                "row": item["row"],

                "col": item["col"],

                "block": item["block"],

                "energy": energy

            })

        ranked.sort(

            key=lambda x: x["energy"],

            reverse=True

        )

        return ranked

    # ----------------------------------------------------
    # Top Blocks
    # ----------------------------------------------------
    def select_blocks(

        self,

        ranked,

        percentage=0.30

    ):
        count = int(
            len(ranked)
            *
            percentage
        )

        if ranked:
            center_row = max(item["row"] for item in ranked) / 2
            center_col = max(item["col"] for item in ranked) / 2
        else:
            center_row = 0
            center_col = 0

        ordered = sorted(
            ranked,
            key=lambda item: (
                abs(item["row"] - center_row) + abs(item["col"] - center_col),
                item["row"],
                item["col"],
            )
        )

        return ordered[:count]

             # ----------------------------------------------------
    # Iterator
    # ----------------------------------------------------
    def coefficient_iterator(
        self,
        blocks
    ):

        for item in blocks:

            block = item["block"]

            for i in range(

                block.shape[0]

            ):

                for j in range(

                    block.shape[1]

                ):

                    yield (

                        item,

                        i,

                        j,

                        block[i, j]

                    )

    # ----------------------------------------------------
    # Clip
    # ----------------------------------------------------
    def clip(self, band):

        return np.clip(

            band,

            0,

            1

        )

    # ----------------------------------------------------
    # Reconstruction
    # ----------------------------------------------------
    def reconstruct_band(

        self,

        coeff,

        band,

        new_band

    ):

        coeff[band] = new_band

        return coeff